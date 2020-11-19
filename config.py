
class Config(object):
    DEBUG = False
    TESTING = False

    FILE_UPLOADS = '/Users/wolfsinem/product-tagging/static/data/uploads'


class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    DEBUG = True

    FILE_UPLOADS = '/Users/wolfsinem/product-tagging/static/data/uploads'
    SESSION_COOKIE_SECURE = False


class TestingConfig(Config):
    pass