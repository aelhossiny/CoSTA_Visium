create_gene_exp_image <- function(input_dir, output_dir){
  
  #Loading libraries
  suppressPackageStartupMessages({
    library(Seurat)
    library(ggplot2)
    library(patchwork)
    library(dplyr)
  })
  
  options(warn=-1)
  
  #Loading sample
  cat(">> Importing dataset\n")
  filename <- list.files(input_dir, pattern = "filtered_feature_bc_matrix.h5")
  dataset <- Load10X_Spatial(data.dir = input_dir, filename = filename)
  
  #Normalizing sample
  cat(">> Normalizing dataset\n")
  dataset <- SCTransform(dataset, assay = "Spatial", verbose = FALSE)
  
  #Finding spatially variable features
  cat(">> Finding variable genes\n")
  ## dataset <- FindSpatiallyVariableFeatures(dataset, assay = "SCT", features = VariableFeatures(dataset),selection.method = "markvariogram")
  ## SpatiallyVariableFeatures <- SPG(dataset, selection.method = "markvariogram")
  HVG <- VariableFeatures(dataset)
  
  #Creating and saving image for each gene
  cat(">> Creating Gene Expression Images\n")
  for (gene in HVG){
    
    
    feature_plot <- 
      
      suppressWarnings({
        suppressMessages({
          
          SpatialFeaturePlot(
            dataset,
            features = gene,
            slot = "data",
            ncol = NULL,
            pt.size.factor = 2,
            alpha = c(1, 1),
            image.alpha = 0,
            stroke = 0) + 
            
            scale_fill_continuous(low = "black",high = "white") + 
            
            theme(plot.background = element_rect(fill = 'black'),
                  panel.grid.major = element_blank(),
                  panel.grid.minor = element_blank(),
                  panel.border = element_blank()) +
            
            NoLegend()
        })
      })
    
    jpeg(filename =  paste0(output_dir,"/",gene,".jpeg"))
    plot(feature_plot)
    dev.off()
    
  }
}

