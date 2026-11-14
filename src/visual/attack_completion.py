"""校验实际完成步数、输入摘要、四阶段证据及最终图片。"""
from src.paths import ProjectPath as Path
from src.visual.image_utils import sha256_file
from src.visual.optimization import normalize_method


def validate_attack_completion(metadata, requested_steps, expected=None):
    if not isinstance(metadata, dict):
        raise ValueError('完成记录必须为对象')
    method = normalize_method(metadata.get('method'))
    if metadata.get('test_fixture') is True or metadata.get('production_backend') is False:
        raise ValueError('离线测试夹具不能登记为正式成图或用于正式评估')
    if metadata.get('status') != 'success':
        raise ValueError('图片方法尚未正常完成')
    output = Path(metadata.get('output_image', ''))
    if not output.is_file() or metadata.get('output_image_sha256') != sha256_file(output):
        raise ValueError('最终图片缺失或摘要改变')
    for key, value in (expected or {}).items():
        if metadata.get(key) != value:
            raise ValueError('完成记录未绑定本次冻结输入：' + key)
    if method == 'ambilage':
        from src.visual._ambilage.evidence import validate_ready
        ready = validate_ready(output.parent, require_production=True)
        if type(requested_steps) is not int or requested_steps < 1 or ready['steps_requested'] != requested_steps:
            raise ValueError('完成凭证实际步数与本次请求不一致')
        if ready['output_image_sha256'] != metadata['output_image_sha256']:
            raise ValueError('完成凭证未绑定本次最终图片')
        if metadata.get('shadow_pool_used') is not False:
            raise ValueError('完整方法不得读入影子池或私有共享记忆池')
        for key in ('source', 'target'):
            if ready.get(key + '_sha256') != metadata.get(key + '_image_sha256'):
                raise ValueError('四阶段证据与本次公开图对不符')
        return
    attack = metadata.get('attack', {})
    if type(requested_steps) is not int or requested_steps < 1:
        raise ValueError('请求步数必须为正整数')
    if (type(attack.get('steps_completed')) is not int
            or attack['steps_completed'] != requested_steps
            or attack.get('steps_requested', requested_steps) != requested_steps
            or attack.get('stop_reason') not in (None, 'requested_steps_completed')):
        raise ValueError('像素优化未完成请求日程')
    if any(type(attack.get(k, 0)) is not int or not 0 <= attack.get(k, 0) <= requested_steps
           for k in ('best_step', 'selected_step')):
        raise ValueError('像素优化选择步无效')
