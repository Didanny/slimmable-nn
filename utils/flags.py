class Flags(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating the object')
            cls._instance = super(Flags, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance