from utils import process

dataset = 'cora'
adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)

# process.plot_eigenvector(adj,2700)
# process.save_as_mat(adj,2700)
process.plot_graph(adj)



# adj_ = process.normalize_adj(adj + sp.eye(adj.shape[0]))
# features, _ = process.preprocess_features(features)

# process.plot_graph(adj)