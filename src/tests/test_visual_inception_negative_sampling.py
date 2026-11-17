"""Micro-only public negative-role and stable selection contracts."""
import copy
from types import SimpleNamespace

import pytest
import torch

from src.visual._visual_inception.negative_sampling import (
    build_negative_policy, select_hard_negatives, validate_negative_selection_audit,
)


def config(count=6,quota=1):
    return SimpleNamespace(hard_negative_count=count,hard_negative_same_category_count=quota,
        hard_negative_proxy_memory_count=quota,hard_negative_popularity_count=quota)


def rows(count=9):
    return [dict(id=f'explicit_fixture_{i}',category='shampoo' if i<3 else 'jeans',reserve_rank=i+1)
            for i in range(count)]


def test_proxy_centrality_is_static_and_not_sales_or_reserve_rank():
    vectors=torch.tensor([[1.,0.],[.9,.1],[.9,-.1],[-1.,0.],[0.,1.],[0.,-1.],[.8,.2],[-.8,.2],[.1,.9]])
    policy=build_negative_policy(rows(),vectors,'SHAMPOO')
    assert policy['popularity']['method']=='query_independent_cosine_centrality_proxy'
    assert policy['popularity']['true_public_popularity_used'] is False
    assert policy['popularity']['sales_popularity_claimed'] is False
    assert len(policy['role_indices']['popular'])==3
    assert policy['role_indices']['same_category']==[0,1,2]
    assert policy['role_indices']['proxy_user_memory']==list(range(9))
    assert policy==build_negative_policy(rows(),vectors.double(),'shampoo')


def test_real_public_metric_requires_source_and_is_preferred_when_present():
    metadata=rows()
    for i,row in enumerate(metadata):
        row['public_popularity']=dict(value=i,metric='public_review_count',source='https://example.org/public-records')
    policy=build_negative_policy(metadata,torch.ones(9,2),'shampoo')
    assert policy['popularity']['true_public_popularity_used'] is True
    assert policy['role_indices']['popular']==[8,7,6]
    assert policy['popularity']['sales_popularity_claimed'] is False
    for row in metadata:
        row['public_popularity'].pop('source')
    assert build_negative_policy(metadata,torch.ones(9,2),'shampoo')['popularity']['true_public_popularity_used'] is False


def test_quota_selection_is_unique_stable_and_auditable_with_overlapping_roles():
    negatives=torch.ones(9,2,dtype=torch.float64)
    policy=build_negative_policy(rows(),negatives,'shampoo')
    indices,audit=select_hard_negatives(torch.ones(2,2,dtype=torch.float64),negatives,policy,config())
    assert indices.tolist()==[list(range(6))]*2
    assert indices.dtype==torch.long and not indices.requires_grad
    assert audit['role_order']==['same_category','popular','proxy_user_memory']
    assert audit['per_query'][0]['role_selected_counts']==dict(same_category=1,popular=1,proxy_user_memory=1)
    assert audit['per_query'][0]['global_fill_count']==3
    assert validate_negative_selection_audit(audit,policy,config(),2)
    changed=copy.deepcopy(audit); changed['per_query'][0]['selected_indices'][1]=0
    with pytest.raises(ValueError): validate_negative_selection_audit(changed,policy,config(),2)
    changed=copy.deepcopy(audit); changed['per_query'][0]['role_selected_indices']['popular']=[True]
    with pytest.raises(ValueError): validate_negative_selection_audit(changed,policy,config(),2)


def test_hardness_tracks_current_query_and_preserves_device():
    negatives=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    metadata=[dict(id=str(i),category='shampoo') for i in range(4)]
    policy=build_negative_policy(metadata,negatives,'shampoo')
    queries=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
    indices,audit=select_hard_negatives(queries,negatives,policy,config(4))
    assert indices[0,0].item()==0 and indices[1,0].item()==1
    assert indices.device==queries.device
    assert validate_negative_selection_audit(audit,policy,config(4),2)


def test_small_pool_reports_role_and_global_shortage_without_duplication():
    metadata=[dict(id='a',category='shampoo'),dict(id='b',category='jeans')]
    negatives=torch.eye(2)
    policy=build_negative_policy(metadata,negatives,'shampoo')
    indices,audit=select_hard_negatives(torch.ones(1,2),negatives,policy,config(32,8))
    assert indices.shape==(1,2) and len(set(indices[0].tolist()))==2
    assert audit['candidate_shortfall']==30
    assert any(audit['per_query'][0]['role_shortfall'].values())
    assert validate_negative_selection_audit(audit,policy,config(32,8),1)


def test_policy_tampering_and_invalid_configuration_stop():
    negatives=torch.ones(9,2); policy=build_negative_policy(rows(),negatives,'shampoo')
    bad=copy.deepcopy(policy); bad['role_indices']['popular']=[8]
    with pytest.raises(ValueError,match='摘要'): select_hard_negatives(torch.ones(1,2),negatives,bad,config())
    with pytest.raises(ValueError,match='配额'): select_hard_negatives(torch.ones(1,2),negatives,policy,config(2))
