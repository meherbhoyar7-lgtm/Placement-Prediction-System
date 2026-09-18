from sklearn.ensemble import GradientBoostingClassifier


def create_model():

    model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.5,)

    return model
