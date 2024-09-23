#########################################################
#Princess Margaret Cancer Research Tower
#Schwartz Lab
#Javier Ruiz Ramirez
#September 2024
#########################################################
#This is a Python script to produce TMC trees using
#the original too-many-cells tool.
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7439807/
#########################################################
#Questions? Email me at: javier.ruizramirez@uhn.ca
#########################################################
import os
import subprocess
import pandas as pd
import matplotlib as mpl
from typing import Union
from typing import Optional

class TMCHaskell:

    #=====================================
    def __init__(
            self,
            output: str,
            tmc_tree_path: str,
            matrix_path: Optional[str] = "",
            list_of_genes: Optional[list] = [],
            use_threshold: Optional[bool] = False,
            high_low_colors: Optional[list] = ["purple",
                                               "red",
                                               "blue",
                                               "aqua"],
            gene_colors: Optional[list] = [],
            annotation_colors: Optional[list] = [],
            method: Optional[str] = "MadMedian",
            tree_file_name: Optional[str] = "tree.svg",
            threshold: Optional[float] = 1.5,
            saturation: Optional[float] = 1.5,
            output_folder: Optional[str] = "tmc_haskell",
            feature_column: Optional[str] = "1",
            draw_modularity: Optional[bool] = False,
            path_to_cell_annotations: Optional[str] = "",
            draw_node_numbers: Optional[bool] = False,
        ):

        self.use_threshold = use_threshold

        if use_threshold:
            if len(list_of_genes) == 0:
                txt  = "List of genes cannot be empty."
                raise ValueError(txt)
            else:
                n_colors = len(gene_colors)
                n_genes = len(list_of_genes)

                if 0 < n_colors and n_colors != 2**n_genes:
                    txt  = "We need 2**n_genes colors."
                    raise ValueError(txt)

        self.feature_column = feature_column
        self.draw_node_numbers = draw_node_numbers
        self.draw_modularity = draw_modularity

        #High-High, High-Low, Low-High, Low-Low.
        # self.high_low_colors = ['purple',
        #                         'red',
        #                         'blue',
        #                         'aqua']

        self.high_low_colors = high_low_colors

        self.high_low_colors_hex = []

        self.gene_colors_hex = []


        # self.list_of_genes = ["FAP","TCF21"]
        # self.list_of_genes = ["FAP"]

        self.list_of_genes = list_of_genes

        # self.gene_colors = ["blue", "red"]
        self.gene_colors = gene_colors

        # self.matrix_path = "./tmci_mtx_data"
        self.matrix_path = matrix_path

        # self.tmc_tree_path = "./data_to_regenerate_tree"
        self.tmc_tree_path = tmc_tree_path

        self.method = method
        self.threshold = threshold


        self.all_genes = []

        self.draw_node_numbers = False

        x = path_to_cell_annotations
        self.path_to_cell_annotations = x

        self.output = os.path.join(output,
                                   output_folder)

        # print(self.output)
        os.makedirs(self.output, exist_ok=True)

        self.saturation = saturation


        self.tree_output = tree_file_name

        self.cluster_path = os.path.join(self.output,
                                         "clusters.csv")
        
        self.annotation_colors = annotation_colors

        self.list_of_colors = []



    #=====================================
    def create_gene_objects(self):


        if self.use_threshold:

            for gene in self.list_of_genes:
                gene_mod = '\\\"' + gene + '\\\"'
                gene_bicolor = '(' + gene_mod + ', ' 
                gene_bicolor += self.method
                gene_bicolor += ' '
                gene_bicolor += str(self.threshold)
                gene_bicolor += ')'
                self.all_genes.append(gene_bicolor)

            for color in self.high_low_colors:
                color_hex = mpl.colors.cnames[color]
                color_hex = ('\\\"' +
                             color_hex.lower() +
                             '\\\"')
                self.high_low_colors_hex.append(color_hex)

            x = '\"DrawItem (DrawThresholdContinuous '
            self.gene_txt = x
            self.list_of_colors = (
                '[' +
                ','.join(self.high_low_colors_hex) +
                ']')
            self.gene_list = (
                '[' + ','.join(self.all_genes) + ']')

        else:

            for gene, color in zip(self.list_of_genes,
                                   self.gene_colors):
                color_hex = mpl.colors.cnames[color]
                color_hex = ('\\\"' +
                             color_hex.lower() +
                             '\\\"')
                gene_mod = '\\\"' + gene + '\\\"'
                self.gene_colors_hex.append(color_hex)
                self.all_genes.append(gene_mod)


            self.gene_txt = '\"DrawItem (DrawContinuous '
            # print(self.gene_colors_hex)
            self.list_of_colors = (
                '[' + ','.join( self.gene_colors_hex) + ']')
            self.gene_list = (
                '[' + ','.join(self.all_genes) + ']')

        #'\"DrawItem (DrawThresholdContinuous 
        # [(\\\"FH\\\", Exact 0), (\\\"FL\\\", Exact 0)])\"',

        self.gene_txt += self.gene_list
        self.gene_txt += ')\"'

    #=====================================
    def execute_command(self):

        # print(self.gene_txt)

        if self.draw_node_numbers:
            draw_node_numbers = "--draw-node-number"
        else:
            draw_node_numbers = ""

        if self.draw_modularity:
            modularity_flag = "--draw-mark"
            modularity_argument = "MarkModularity"
        else:
            modularity_flag = ""
            modularity_argument = ""

        if os.path.exists(self.path_to_cell_annotations):
            labels_flag = "--labels-file"
            labels_argument = self.path_to_cell_annotations
        else:
            labels_flag = ""
            labels_argument = ""

        if os.path.exists(self.matrix_path):
            matrix_path_flag = "--matrix-path"
            matrix_path_argument = self.matrix_path
        else:
            matrix_path_flag = ""
            matrix_path_argument = ""

        if 0 < len(self.list_of_genes):
            draw_leaf_flag = "--draw-leaf"
            draw_leaf_arguments = self.gene_txt
        else:
            draw_leaf_flag = ""
            draw_leaf_arguments = ""

        if 0 < len(self.list_of_colors):
            draw_colors_flag = "--draw-colors"
            draw_colors_arguments = self.list_of_colors
        else:
            draw_colors_flag = ""
            draw_colors_arguments = ""

        command = ["too-many-cells",
                "make-tree",
                "",
                matrix_path_flag,
                matrix_path_argument,
                "",
                "--prior",
                self.tmc_tree_path,
                "",
                #"--normalization",
                #"UQNorm"
                labels_flag,
                labels_argument,
                "",
                draw_node_numbers,
                "",
                modularity_flag,
                modularity_argument,
                "",
                "--feature-column",
                self.feature_column,
                "",
                #'\"DrawItem (DrawThresholdContinuous
                #[(\\\"FH\\\", Exact 0),
                #(\\\"FL\\\", Exact 0)])\"',
                draw_leaf_flag,
                draw_leaf_arguments,
                "",
                draw_colors_flag,
                draw_colors_arguments,
                "",
                "--draw-scale-saturation",
                str(self.saturation),
                "",
                "--dendrogram-output",
                self.tree_output,
                "",
                "--labels-output",
                "",
                "--output",
                self.output,
                "",
                ">",
                self.cluster_path]
        # print(command)
        command = list(filter(len, command))
        command = " ".join(command)
        #print(">>>>>>>>>>>")
        #print(command)
        p = subprocess.call(command, shell=True)

    #=====================================
    def run(self):

        if 0 < len(self.list_of_genes):
            self.create_gene_objects()
        else:
            for color in self.annotation_colors:
                color_hex = mpl.colors.cnames[color]
                color_hex = ('\\\"' +
                             color_hex.lower() +
                             '\\\"')
                self.list_of_colors.append(color_hex)

        self.execute_command()