import kserve
from typing import Dict


class Model(kserve.Model): 
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]        
        return {"instances": inputs}


if __name__ == "__main__":
    model = Model("custom-model")
    model.load()
    kserve.ModelServer(workers=1, http_port=8080).start([model])
    