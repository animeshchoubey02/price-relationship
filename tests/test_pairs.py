import pandas as pd
from price_ratio.pairs import build_pairs

def test_pair_count_and_target_direction():
    d=pd.DataFrame([{"brand":"A","category":"x|y","price":20,"product_name":"red tool"},{"brand":"A","category":"x|y","price":10,"product_name":"blue tool"},{"brand":"B","category":"x|z","price":5,"product_name":"other"}])
    X,y,e=build_pairs(d)
    assert len(X)==3 and len(y)==3 and len(e)==3
    assert y[0]>0
