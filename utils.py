class dotdict:
    def __init__(self, base_dict):

        self.base_dict = base_dict

    def __getattr__(self, name):
        return self.base_dict[name]

    def __reduce__(self):
        return (self.__class__, (self.base_dict,))