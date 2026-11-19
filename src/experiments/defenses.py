"""对共享池所有图片施加相同防御，查询图片保持原样。"""
import gc
from src.paths import ProjectPath as Path
import io
from PIL import Image, ImageFilter

from src.visual.audit import digest, read_json, save_json
from src.visual.image_utils import sha256_file

DEFENSE_MODES = ('none', 'cognitiveguard_sd21', 'panel', 'panel_with_sd21')


def conditions():
    result = [dict(label='none', kind='none', size=512)]
    result += [dict(label=f'blur{str(r).replace(".", "p")}', kind='blur', radius=r, size=512) for r in (0.5, 1, 2, 4)]
    result += [dict(label=f'jpeg{q}', kind='jpeg', quality=q, size=512) for q in (90, 75, 50, 25)]
    result += [dict(label='resize512', kind='resize', size=512), dict(label='vae512', kind='vae', size=512)]
    result += [dict(label=f'diff512_t{t}_s{s}', kind='diffusion', size=512, timestep=t, seed=s)
               for t in (99, 199, 399) for s in range(1001, 1006)]
    return result


def sd21_conditions():
    """独立留出 CognitiveGuard 系统一扩散净化条件（使用严格留出的 SD 2.1-base 净化器）。"""
    result = [dict(label='sd21_vae512', kind='vae', size=512, purifier_key='eval_purifier_sd21')]
    result += [
        dict(label=f'sd21_diff512_t{t}_s{s}', kind='diffusion', size=512, timestep=t, seed=s, purifier_key='eval_purifier_sd21')
        for t in (99, 199)
        for s in range(1001, 1006)
    ]
    return result


def conditions_for_mode(mode='panel'):
    if mode == 'none':
        return conditions()[:1]
    if mode == 'cognitiveguard_sd21':
        return conditions()[:1] + sd21_conditions()
    if mode == 'panel':
        return conditions()
    if mode == 'panel_with_sd21':
        return conditions() + sd21_conditions()
    raise ValueError('未知防御模式：' + str(mode))


def table_groups():
    rows = [('无防御', ['none'])]
    rows += [(f'模糊半径 {r}', [f'blur{str(r).replace(".", "p")}']) for r in (0.5, 1, 2, 4)]
    rows += [(f'压缩质量 {q}', [f'jpeg{q}']) for q in (90, 75, 50, 25)]
    rows += [('缩放重建', ['resize512']), ('变分自编码重建', ['vae512'])]
    rows += [(f'扩散净化起点 {t}', [f'diff512_t{t}_s{s}' for s in range(1001, 1006)]) for t in (99, 199, 399)]
    return rows


def sd21_table_groups():
    rows = [('留出 SD 2.1 变分自编码重建', ['sd21_vae512'])]
    rows += [
        (f'留出 SD 2.1 系统一扩散净化起点 {t}', [f'sd21_diff512_t{t}_s{s}' for s in range(1001, 1006)])
        for t in (99, 199)
    ]
    return rows


def all_table_groups():
    return table_groups() + sd21_table_groups()


class Defender:
    def __init__(self, output, device='cuda'):
        self.output, self.device, self.purifier = Path(output), device, None
        self._purifier_key = None
        self._impl_sha256 = sha256_file(__file__)
        self._sha_cache = {}
        self._purifier_snapshots = {}

    def _cached_sha256(self, file_path):
        p = Path(file_path).resolve()
        st = p.stat()
        cache_key = (str(p), st.st_mtime_ns, st.st_size)
        digest_val = self._sha_cache.get(cache_key)
        if digest_val is None:
            digest_val = sha256_file(p)
            self._sha_cache[cache_key] = digest_val
        return digest_val

    def _purifier_config(self, purifier_key='purifier'):
        from src.visual.diffusion_purification import PurificationConfig
        if purifier_key == 'purifier':
            return PurificationConfig(working_size=512)
        from src.model_registry import resolve_model
        return PurificationConfig(model_path=resolve_model(purifier_key), working_size=512)

    def _purifier_snapshot(self, purifier_key='purifier'):
        if purifier_key in self._purifier_snapshots:
            return self._purifier_snapshots[purifier_key]
        import src.visual.diffusion_purification as dp
        # Use disk-only snapshot when real LatentDiffusionPurifier is active so cache hits never load GPU weights.
        if getattr(dp.LatentDiffusionPurifier, '__module__', '') == dp.__name__:
            snap = dp.purifier_snapshot(self._purifier_config(purifier_key))
            self._purifier_snapshots[purifier_key] = snap
            return snap
        self._ensure_purifier(purifier_key)
        snap = self.purifier.snapshot()
        self._purifier_snapshots[purifier_key] = snap
        return snap

    def close(self):
        if self.purifier is not None:
            self.purifier = None
            self._purifier_key = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _ensure_purifier(self, purifier_key='purifier'):
        if self.purifier is not None:
            if self._purifier_key == purifier_key or (self._purifier_key is None and purifier_key == 'purifier'):
                return
            self.close()
        from src.visual.diffusion_purification import LatentDiffusionPurifier
        cfg = self._purifier_config(purifier_key)
        self.purifier = LatentDiffusionPurifier(cfg, device=self.device)
        self._purifier_key = purifier_key

    def __call__(self, image_path, spec):
        if spec['kind'] == 'none':
            return Path(image_path).resolve()
        import PIL
        identity = dict(input_sha256=self._cached_sha256(image_path), condition=spec,
                        implementation_sha256=self._impl_sha256, pillow=PIL.__version__)
        purifier_key = spec.get('purifier_key', 'purifier')
        if spec['kind'] in ('vae', 'diffusion'):
            identity['purifier'] = self._purifier_snapshot(purifier_key)
        path = self.output / spec['label'] / (digest(identity) + '.png')
        sidecar = path.with_suffix('.json')
        if path.exists() or sidecar.exists():
            if not path.is_file() or not sidecar.is_file():
                raise ValueError('防御缓存不完整')
            saved = read_json(sidecar)
            if saved['identity'] != identity or saved['output_sha256'] != self._cached_sha256(path):
                raise ValueError('防御缓存图片或身份已改变')
            return path
        if spec['kind'] in ('vae', 'diffusion'):
            self._ensure_purifier(purifier_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as opened:
            image = opened.convert('RGB')
        if spec['kind'] == 'blur':
            image.filter(ImageFilter.GaussianBlur(spec['radius'])).save(path)
        elif spec['kind'] == 'jpeg':
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=spec['quality'], subsampling=2)
            stream.seek(0)
            with Image.open(stream) as decoded:
                decoded.convert('RGB').save(path)
        elif spec['kind'] == 'resize':
            from src.visual.diffusion_purification import working_shape
            h, w = working_shape((image.height, image.width), spec['size'])
            image.resize((w, h), Image.Resampling.BICUBIC).resize(image.size, Image.Resampling.BICUBIC).save(path)
        elif spec['kind'] in ('vae', 'diffusion'):
            import torch
            from src.visual.image_utils import read_tensor, save_tensor_image
            tensor = read_tensor(image_path, self.purifier.device)
            with torch.no_grad():
                output = (self.purifier.reconstruct(tensor) if spec['kind'] == 'vae'
                          else self.purifier(tensor, seed=spec['seed'], start_timestep=spec['timestep']))
            save_tensor_image(path, output)
        else:
            raise ValueError('未知防御设置')
        save_json(sidecar, dict(identity=identity, output_sha256=sha256_file(path)))
        return path
