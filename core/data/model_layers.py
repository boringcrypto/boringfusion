import hashlib
from collections import OrderedDict
import torch
from core.safe_unpickler import torch_load

class ModelLayers(OrderedDict):
    def __init__(self, name="", layer_count=0) -> None:
        self.name = name
        self.layer_count = layer_count
        super().__init__()

    @property
    def dtypes(self):
        types = [str(layer.dtype) for layer in list(self.values())]
        return str.join(", ", [str(dtype) for dtype in set(types)])

    @property
    def version_hash(self):
        total = torch.zeros(256)
        for key in self.keys():
            flat = self[key].flatten()
            length = flat.size(0)
            if (length<=256):
                total[0:length] += flat
            else:
                total.add_(flat[length // 2 - 128 : length // 2 + 128])
        hash = hashlib.md5(str([e.item() for e in total]).encode()).hexdigest()[:8]

        return hash

    @property
    def content_hash(self):
        total = torch.zeros(256)
        for key in self.keys():
            flat = self[key].flatten()
            length = flat.size(0)
            if (length<=256):
                total[0:length] += flat.half()
            else:
                total.add_(flat[length // 2 - 128 : length // 2 + 128].half())
        hash = hashlib.md5(str([e.item() for e in total]).encode()).hexdigest()[:8]

        return hash

    @property
    def parameter_count(self):
        return sum(torch.numel(layer) for layer in self.values())

    def set(self, layers):
        if len(layers) == self.layer_count:
            self.clear()
            self.update(layers)
        else:
            print(self.name, "model has the wrong number of layers, skipped", len(layers), self.layer_count)

    def set_from_state_dict(self, state_dict, base_key="", mapping=None, prepend=""):
        keys = [key for key in state_dict.keys() if not len(base_key) or key.startswith(base_key)]
        layers = {prepend + mapping[key[len(base_key):]] if mapping else prepend + key[len(base_key):] : state_dict[key] for key in keys}
        
        # Sort by key to ensure order is always the same to generate the same hash
        self.set({key : layers[key] for key in sorted(layers.keys())})

    def half_(self):
        """Converts all layers to half precision (fp16)
        """
        for key in self.keys():
            self[key] = self[key].half()

    def save(self, filename):
        torch.save({
            "name": self.name,
            "layer_count": self.layer_count,
            "state_dict": { key:value for key, value in self.items() },

            # This meta-data is stored so we don't have to load the full data to get this info
            # Once loaded, this info should be ignored and the actual class properties used
            "content_hash": self.content_hash,
            "version_hash": self.version_hash,
            "dtypes": self.dtypes,
            "parameter_count": self.parameter_count
        }, filename)

    @classmethod
    def load(cls, filename):
        data = torch_load(filename, map_location="cpu")
        model_layers = cls(data["name"], data["layer_count"])
        model_layers.set_from_state_dict(data["state_dict"])

        return model_layers

    def merge_(self, model, weight=0.5):
        """Merge with another model (in place)

        Args:
            model (dict or ModelLayers): Another model with the same keys and shapes
            weight (float, optional): The weight of the model being merged in. A weight of one doesn't do anything, weight of 1 overwrites with the model. Defaults to 0.5.
        """
        for key in self.keys():
            self[key] = self[key] * (1 - weight) + model[key] * weight

        return self

    def merge(self, model, weight=0.5):
        """Merge with another model and returns the merged ModelLayers

        Args:
            model (dict or ModelLayers): Another model with the same keys and shapes
            weight (float, optional): The weight of the model being merged in. A weight of one doesn't do anything, weight of 1 overwrites with the model. Defaults to 0.5.
        """
        combined = ModelLayers(self.name, self.layer_count)
        for key in self.keys():
            combined[key] = self[key] * (1 - weight) + model[key] * weight
        
        return combined

    def add_diff_(self, base_model, trained_model, weight=1):
        for key in self.keys():
            self[key] = self[key] + (trained_model[key] - base_model[key]) * weight

        return self

    def add_diff_(self, base_model, trained_model, weight=1):
        combined = ModelLayers(self.name, self.layer_count)
        for key in self.keys():
            combined[key] = self[key] + (trained_model[key] - base_model[key]) * weight

        return combined

    def __str__(self):
        if len(self):
            return self.name + ": " + self.content_hash + ":" + self.version_hash + " - " + self.dtypes + " - " + str(self.parameter_count // 1000000) + "M params"
        else:
            return self.name + ": None"
