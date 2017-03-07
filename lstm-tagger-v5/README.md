## lstm-tagger-v5
Thử sử dụng một số kết hợp khác nhau giữa word-embedding vector và feature-embedding vector 

```python
ttt = None
for ilayer in range(len(self.feature_maps)):
    f = self.feature_maps[ilayer]
    # input_dim += f['dim']
    af_layer = EmbeddingLayer(len(f['id_to_ftag']), f['dim'], name=f['name'] + '_layer')
    f_layers.append(af_layer)
    temp = af_layer.link(features_ids[ilayer])  
    input_dim = input_dim + 100
    inputs.append(temp * word_input)
```

