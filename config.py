
class Config(object):
    DEBUG = False
    TESTING = False

    IMAGE_UPLOADS = '/Users/wolfsinem/product-tagging/static/img/uploads'


class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    DEBUG = True

    IMAGE_UPLOADS = '/Users/wolfsinem/product-tagging/static/img/uploads'
    SESSION_COOKIE_SECURE = False


class TestingConfig(Config):
    pass