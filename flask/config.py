class Config:
    PORT = 5000
    HOST = "0.0.0.0"


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = dict(
    dev=DevelopmentConfig,
    prod=ProductionConfig
)
