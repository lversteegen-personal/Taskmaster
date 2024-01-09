import keras

class dotdict:
    def __init__(self, base_dict):

        self.base_dict = base_dict

    def __getattr__(self, name):
        return self.base_dict[name]

    def __reduce__(self):
        return (self.__class__, (self.base_dict,))

#There seems to be no reasonable way to copy a model including its optimizer
def copy_network(network):

    network.save("temp.h5")
    return keras.models.load_model("temp.h5")
    
