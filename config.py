
class Config(object):
    DEBUG = False
    TESTING = False

    FILE_UPLOADS = '/Users/wolfsinem/product-tagging/static/data/uploads'
    FILE_EXPORTS = '/Users/wolfsinem/product-tagging/static/data/exports'
    ALLOWED_FILE_EXTENSIONS = ["CSV"]

class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    DEBUG = True

    FILE_UPLOADS = '/Users/wolfsinem/product-tagging/static/data/uploads'
    FILE_EXPORTS = '/Users/wolfsinem/product-tagging/static/data/exports'
    SESSION_COOKIE_SECURE = False
    ALLOWED_FILE_EXTENSIONS = ["CSV"]


class TestingConfig(Config):
    pass