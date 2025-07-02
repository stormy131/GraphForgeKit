from utils.metrics import haversine_dist, euclid_dist, cosine_dist, mahalanobis_dist


METRICS = {
    "Haversine"     : haversine_dist,
    "Euclidean"     : euclid_dist,
    "Cosine"        : cosine_dist,
    "Mahalonobis"   : mahalanobis_dist,
}
