"""End-to-end pairwise learning and graph reconciliation."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from .data import load_catalog
from .pairs import build_pairs
from .graph import reconcile

def run(path, random_state=42, max_pairs=20000, anchors=None):
    catalog=load_catalog(path)
    X,y,edges=build_pairs(catalog,max_pairs=max_pairs,random_state=random_state)
    idx=np.arange(len(y)); train,test=train_test_split(idx,test_size=.25,random_state=random_state)
    model=RandomForestRegressor(n_estimators=300,min_samples_leaf=2,max_features=.85,n_jobs=-1,random_state=random_state)
    model.fit(X.iloc[train],y[train])
    pred=model.predict(X.iloc[test])
    all_pred=model.predict(X)
    first_pass_prices=reconcile(len(catalog),edges,all_pred,anchors=anchors)

    # Relearn the attribute-to-ratio mapping from the reconciled structure.
    # This makes the structural correction part of learning, not only a post-process.
    reconciled_targets=first_pass_prices[edges[:,0]]-first_pass_prices[edges[:,1]]
    second_model=RandomForestRegressor(n_estimators=300,min_samples_leaf=2,max_features=.85,n_jobs=-1,random_state=random_state+1)
    second_model.fit(X,reconciled_targets)
    second_pass_pred=second_model.predict(X)
    log_prices=reconcile(len(catalog),edges,second_pass_pred,anchors=anchors)

    observed=np.log(catalog.price.to_numpy())
    # Center both vectors: absolute price scale is not identifiable from ratios alone.
    centered_observed=observed-observed.mean(); centered_est=log_prices-log_prices.mean()
    edge_residual=second_pass_pred-(log_prices[edges[:,0]]-log_prices[edges[:,1]])
    return {"catalog":catalog,"model":model,"relearned_model":second_model,"features":X.columns.tolist(),"pair_count":len(y),
      "pairwise_test_mae":float(mean_absolute_error(y[test],pred)),
      "pairwise_test_r2":float(r2_score(y[test],pred)) if len(test)>1 else float("nan"),
      "graph_edge_rmse":float(np.sqrt(np.mean(edge_residual**2))),
      "centered_log_price_rmse":float(np.sqrt(np.mean((centered_observed-centered_est)**2))),
      "estimated_relative_prices":np.exp(log_prices-log_prices.mean())}
