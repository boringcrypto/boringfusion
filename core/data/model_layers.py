import os
import hashlib
from collections import OrderedDict
import torch
from core.safe_unpickler import torch_load

class ModelLayersInfo:
    def __init__(self, shortname, model, name, filename, content_hash, version_hash, dtypes, parameter_count, info_url, download_url, description, found_in):
        self.shortname = shortname
        self.model = model
        self.name = name
        self.filename = filename
        self.content_hash = content_hash
        self.version_hash = version_hash
        self.dtypes = dtypes
        self.parameter_count = parameter_count
        self.info_url = info_url
        self.download_url = download_url
        self.description = description
        self.found_in = found_in

    def __repr__(self):
        return f"""ModelLayersInfo(
        "{self.shortname}",
        "{self.model}",
        "{self.name}",
        "{self.filename}",
        "{self.content_hash}",
        "{self.version_hash}",
        "{self.dtypes}",
        "{self.parameter_count}",
        "{self.info_url}",
        "{self.download_url}",
        "{self.description}",
        {self.found_in},
    )"""


class ModelMap(dict):
    def __call__(self, search: str):
        if search in self:
            return self[search]

        result = [model for model in self.values() if model.content_hash == search or model.shortname == search or model.name == search]
        if len(result):
            return result[0]

    def __repr__(self):
        return "ModelMap({\n" + str.join(",\n", ['    "' + key + '": ' + str(value) for key, value in self.items()]) + "\n})"


class ModelLayers(OrderedDict):
    def __init__(self, name="", layer_count=0) -> None:
        self.name = name
        self.layer_count = layer_count
        self.info = ModelLayersInfo("", name, "", "", "", "", "", 0, "", "", "", [])
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

    def _set(self, layers):
        self.clear()
        self.update(layers)

        return self

    def set(self, layers):
        if len(layers) == self.layer_count:
            self._set(layers)
        else:
            print(self.name, "model has the wrong number of layers, skipped", len(layers), self.layer_count)

        return self

    def set_from_state_dict(self, state_dict, base_key="", mapping=None, prepend=""):
        keys = [key for key in state_dict.keys() if not len(base_key) or key.startswith(base_key)]
        layers = {prepend + mapping[key[len(base_key):]] if mapping else prepend + key[len(base_key):] : state_dict[key] for key in keys}
        
        # Sort by key to ensure order is always the same to generate the same hash
        self.set({key : layers[key] for key in sorted(layers.keys())})

        return self

    def half(self):
        """Converts all layers to half precision (fp16) and returns a new ModelLayers
        """       
        return ModelLayers(self.name, self.layer_count)._set({key:self[key].half() for key in self.keys()})

    def half_(self):
        """Converts all layers to half precision (fp16) in place
        """
        for key in self.keys():
            self[key] = self[key].half()

        return self

    def cuda(self):
        """Converts all layers to half precision (fp16) and returns a new ModelLayers
        """       
        return ModelLayers(self.name, self.layer_count)._set({key:self[key].cuda() for key in self.keys()})

    def cuda_(self):
        """Converts all layers to half precision (fp16) in place
        """
        for key in self.keys():
            self[key] = self[key].cuda()

        return self

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
    def load(cls, path_name_or_modellayersinfo: str or ModelLayersInfo, device="cpu"):
        info = None

        # Assume it's the filename
        filename = path_name_or_modellayersinfo
        if isinstance(path_name_or_modellayersinfo, str):
            if not os.path.exists(path_name_or_modellayersinfo):
                # If not a path, try to load the model_map and search it for a match
                try:
                    from model_map import map
                    filename = map(path_name_or_modellayersinfo)
                except: 
                    print("No model map found")
            
        # If a ModelLayersInfo was passed in or found by search, get the filename
        if isinstance(filename, ModelLayersInfo):
            info = filename
            filename = filename.filename

        data = torch_load(filename, map_location=device)
        model_layers = cls(data["name"], data["layer_count"])
        if info:
            model_layers.info = info
        model_layers._set(data["state_dict"])

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


