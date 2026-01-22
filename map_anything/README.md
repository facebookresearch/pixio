# MapAnything Model With Pixio Encoder

We provide a pre-trained model with Pixio encoder. This model is trained with the same recipe of MapAnything model, but with pixio-vith16 encoder.
load model with:

```python
from map_anything.model import MapAnything

model = MapAnything.from_pretrained("facebook/pixio-vith16-mapanything")
```
