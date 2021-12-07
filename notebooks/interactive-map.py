
import numpy as np
import umap.umap_ as umap
import umap.plot
import umap.umap_ as umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)

# load dataset and y labels for coloring groups
y_pred_acinar = np.load("y_pred_acinar.npy")#[0:5] # uncomment slicing to speed up for testing
dataset = np.load("acinar_cell_carcinoma.npy")#[0:5]
au_tar_acinar = np.load("au_tar_acinar.npy")#[0:5]
gene_names = np.load("gene_list.npy")
embedding_after_acinar = umap.UMAP(n_neighbors=5, min_dist=0.0, n_components=2,
			metric='correlation').fit_transform(y_pred_acinar)
embedding_after_acinar_df = pd.DataFrame(embedding_after_acinar)
embedding_after_acinar_df.columns=['UMAP1','UMAP2']
embedding_after_acinar_df["new"]=np.argmax(au_tar_acinar, axis = 1)

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
data_colors = [colors[embedding_after_acinar_df["new"][i]] for i in range(0, len(embedding_after_acinar_df))]

fig, ax = plt.subplots(figsize=(7, 7))
sc = plt.scatter(embedding_after_acinar_df["UMAP1"], embedding_after_acinar_df["UMAP2"], s=5, c=data_colors, alpha=1)



# interaction globals
annotations = {10000:0}
annotations.clear()
ab = None
keep_labels = False

def update_annot(ind):
#	print("update annotation")
	index = ind["ind"][0]
	
	if index in annotations:
		return 

	# caption to show next to image
	class_num = embedding_after_acinar_df["new"][index]
	caption = "[" + str(class_num) + "] {}".format(gene_names[class_num])
	pos = sc.get_offsets()[index]
	
	annot = ax.annotate("", xy=(0,0), xytext=(30,30),
						textcoords="offset points",
						bbox=dict(boxstyle="round", fc="w"),
						arrowprops=dict(arrowstyle="->", color=data_colors[index]), size=8)
	annotations[index] = annot
	annot.xy = pos
	annot.set_text(caption)
	annot.get_bbox_patch().set_alpha(0.9)	
	annot.set_visible(True)
	
	im = Image.fromarray(np.uint8(dataset[index]))
	imagebox = OffsetImage(im, zoom=0.8)
	imagebox.image.axes = ax
	ab = AnnotationBbox(imagebox, pos,
						xybox=(62, -14),
						xycoords='data',
						boxcoords="offset points",
						pad=0)
	ab.set_alpha(0.5)	
	ax.add_artist(ab)
	ab.set_zorder(100)

def hover(event):
	global keep_labels, saved_artists, saved_indices
#	if keep_labels:
#		return
	
	def remove():
		for idx in annotations:
			if not idx in saved_indices:
				annotations[idx].set_visible(False)
				del annotations[idx]
#		annotations.clear()
#		print(len(ax.artists), ax.artists)
		for i in range(len(ax.artists) - 1, -1, -1):
			if not i in saved_artists:
#				print("deleting artist: ", i, saved_artists)
				del ax.artists[i]
			
		fig.canvas.draw_idle()
		
	remove()
	if event.inaxes == ax:
		cont, ind = sc.contains(event)
		if cont:
			update_annot(ind)
			fig.canvas.draw_idle()
		else:
			remove()

saved_indices = set()
saved_artists = set()
# click on a point to lock in an image or in black space to clear
def onclick(event):
	global keep_labels
	if event.inaxes == ax:
		cont, ind = sc.contains(event)
		if cont:
			saved_indices.add(ind["ind"][0])
			saved_artists.add(len(ax.artists) - 1)
			keep_labels = True
			update_annot(ind)
			for idx in annotations:
				annotations[idx].set_visible(True)
				
			fig.canvas.draw_idle()
		else:
			keep_labels = False
			for idx in annotations:
				annotations[idx].set_visible(False)
			annotations.clear()
			saved_indices.clear()
			saved_artists.clear()
			ax.artists = list()
			
			fig.canvas.draw_idle()
			

plt.title("Clusters after training")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
fig.canvas.mpl_connect("motion_notify_event", hover)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()