
## Dataset Setting
### M21-404 sample WSIs
The dataset can be accessed by 
#### WSI images
CRC [here](https://abbvie.sharepoint.com/:x:/r/teams/cdxpathology/_layouts/15/Doc.aspx?sourcedoc=%7B5C2D8776-58BA-417C-9DB6-D22D56FE5209%7D&file=FINAL_Reconciled_SampleID_clinical_NSCLC_GEA.xlsx&action=default&mobileredirect=true&wdsle=0) and
NSCLC_GEA [here](https://abbvie.sharepoint.com/:x:/r/teams/cdxpathology/_layouts/15/Doc.aspx?sourcedoc=%7BA7AB4864-8151-4956-90D0-2D8CE1B33B0D%7D&file=FINAL_Reconciled_sampleID_clinical_CRC.xlsx&action=default&mobileredirect=true) by the column `HE Concentriq Image Storage Key`
#### CRC tumor mask
[Sample id and H&E id match](https://abbvie.sharepoint.com/:x:/r/teams/cdxpathology/_layouts/15/Doc.aspx?sourcedoc=%7B0DA6AAA7-18BD-4A72-B4E6-A3E08B77280A%7D&file=CRC_CLI_M21-404-HE_harmonized_feature_table.xlsx&action=default&mobileredirect=true)


## Data folder structure

```
├── requirement.txt
├── get_data
│   └── data_sheets
│        └── FINAL_Reconciled_sampleID_clinical_CRC.xlsx
│        └── ...
│        └── image_id_res.csv # aggregated wsi info
│   └── concentriq
│   └── api_get_data.ipynb #aggregate datasheet
│   └── crc_h5.ipynb #aggregate h&e mask info for crc
│   └── 20240719 #Tumor mask
│└── model
│   └── construct_graph.ipynb
│   └── tumor_only_data.ipynb
│   └── three_layer_train_model.ipynb
│   └── GNN.py
│   └── single_layer_train_model.ipynb
│   └── crc_tumor_train.ipynb
│   └── id_list.h5
├── prov-gigapath
├── patch_embeddings
│   └── 20XEmbeddings
│        └── id_slide_tensor.pt
│        └── id_tile_tensor.pt
│   └── 20XTiles
│   └── 20Xslide_embeding.h5
│   └── patch_gigapath.ipynb
│   └── tile_umap.ipynb
├── utils
│   └── conostant.py
│   └── slide_utils.py
│   └── train_utils.py
│   └── utils_fun.py
```
## Requirements
Required modules can be installed via requirements.txt under the project root
```
pip install -r requirements.txt
```
## 1. Metadata and images
```
# go to /get_data/api_get_data.ipynb
# Save the organized data
df.to_csv(sheets_directory+'image_id_res.csv')
# go to /get_data/crc_h5.ipynb merge tumor mask data
image_id_res_df.to_csv(image_id_res)
# csv file column: 0,images_id,cancer,subj_id,ORR,BOR,3 class,storageKey,HE_id
```

## 2. Tissue segmentation and tiling 
We Used `Pyfast` to 
Install it refer to [Pyfast](https://fast.eriksmistad.no/install-ubuntu-linux.html#python-linux) and learn it by [tutorial](https://fast.eriksmistad.no/python-tutorial-wsi.html).


## 3. Embeddings generations
We Used `prov-gigapath`
1. https://github.com/prov-gigapath/prov-gigapath
`git clone https://github.com/prov-gigapath/prov-gigapath.git`. Follow the README for prov-gigapath to create env `gigapath` and install the requirement for gigapath. Activate the env for all the following task `conda activate gigapath`
2. Model Download replace your `HF_TOKEN` 
```
# go to /patch_embeddings/patch_gigapath.ipynb
os.environ["HF_TOKEN"] = "Your TOKEN"
```
3. tile and embedding
```
# go to /patch_embeddings/patch_gigapath.ipynb, run the code
# The tile images, embeddings are saved in dir nXTiles and nXEmbeddings
# csv file column: 0,images_id,cancer,subj_id,ORR,BOR,3 class,storageKey,HE_id
```
4. aggregate the tile level and slide level embedding for all wsi in one file 
```
# in /patch_embeddings/tile_umap.ipynb
# First several block
concate_slide_Embeddings(pt_file_path, file_name)
concate_X_Embeddings(pt_file_path, file_name)
# Saved the nXslide_embeding.h5, nXtile_embeding.h5 under /patch_embeddings/
```
## 4. Clustering of embeddings

Unsupervised kmeans cluster for all the tiles from all slide
Check [faiss](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization) for GPU-based kmeans 
```
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```
Used faiss to get 64 clusters draw Umap and mannuly identified the clusters
```
# go to \patch_embeddings\tile_cluster_64.ipynb
# Then draw Umap \patch_embeddings\draw_umap.ipynb
```
The results of cluster is storage and annotated on the [slide](https://abbvie.sharepoint.com/:p:/t/cdxpathology/EWOzROmn1WlCmcGQag1mbJoBg-t2anqNdrW3o9kUAgRPUQ?e=yh7ejb). Then the annotation is as `\patch_embeddings\cluster_is_include.xlsx`
The final embedding with cluster and if_include is in `\patch_embeddings\20Xtile_w_64cluster_fix.h5`

## 5. Creating graphs
Contruct the graph, node is the tile embeddings, and edge is the 8 near neighbor
```
# go to \model\construct_graph.ipynb
```
- Three Layer graph
```
with open('three_layer_graph.pkl', 'wb') as f:
    pickle.dump(data_list, f)
#output stucture: a list of graphs of all samples [[(graph5x, graph10x, graph20x)], ...] for each sample: (graph_5x, graph_10x, graph_20x), graphNx: [Data(x=[47, 1536], edge_index=[2, 110], y=[1], coordinates=[47, 2])] 
```
- Tumor only 
```
with open('crc_20x_tumor_graph.pkl', 'wb') as f:
    pickle.dump(data_list, f)
```


## 6. Training the GNN – modules required, environment needed
Install lib [pyg](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) based on your machine
- Three Layer graph
```
# Go to \model\three_layer_train_model.ipynb
```
- CRC Tumor only 
```
# Go to \model\tumor_only_data.ipynb to extract the tumor mask for tiles
crc_tile_df.to_hdf('20x_crc_embeddings.h5', key='data')
# Go to \model\crc_tumor_train.ipynb
```
## 7. Identifying important tiles, highlighting them in graphs
- Tumor only 
```
# Go to \model\crc_tumor_train.ipynb
show_slide_info_with_graph(slide_path, data_20x.cpu(), data_20x.coordinates.cpu(), highlight_nodes=highlight_nodes, level=2)
```
## 8. Overlay of clusters on graphs.
Use `is_include` column in `20x_crc_embeddings.h5` to filter the tile
