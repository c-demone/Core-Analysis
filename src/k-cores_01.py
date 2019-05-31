#!/usr/bin/env python

#Developped by Christopher Demone. February 2019.

import os
import sys
from collections import defaultdict
from pandas import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mayavi import mlab
import random
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from math import sqrt, cos, sin, pi


"""Program to find K-Cores of a graph and generate corresponding 
   adjacency matrix. The code can also perform core decomposition.
   Depencies not included by default in anaconda:
   (1) Networkx: pip install --user networkx
   (2) Mayavi: pip install --user mayavi
"""

class GraphGen:
    """This class represents an undirected graph using adjacency
       list representation.
    """

    def __init__(self, vertices, fname):
        self.V = vertices               # No. of vertices
        self.graph = defaultdict(list)  # Default dictionary to store graph
        self.file_ = fname              # Name of adjacency matrix to read from
        # Only enough colors for eccentricity of 1-5.
        # Also find a way to add the color bar directly to plot.
        #self.eccColors = ['#009999', '#800060', '#1a75ff', '#e60073', '#ff0000'] 
        
        self.eccColors = ["#1e88e5", '#d81b60', '#8e24aa', '#3949ab', '#E53935']

        #self.eccColors = ["#3949ab", "#5e35b1", "#8e24aa", "#d81b60", "#e53935"]
        
        self.shlColors = ["#f44336", "#b71c1c", "#ad1457", "#e91e63", "#4a148c", 
                          "#9c27b0", "#311b92", "#673ab7", "#1a237e", "#3f51b5", 
                          "#0d47a1", "#1976d2", "#01579b", "#0277bd", "#006064",
                          "#0097a7", "#004d40", "#00796b", "#1b5e20", "#388e3c",
                          "#33691e", "#7cb342", "#f57f17", "#c0ca33", "#ff6f00", 
                          "#fbco2d", "#ff6f00", "#e65100", "#f89800", "#bf360c", 
                          "#f4511e", "#795548", "#757575", "#455a64", "#37474f", 
                          "#263238", "#212121"]
        
        self.textFont = {'family': 'sans-serif',
                         'color': 'black',
                         'weight': 'bold',
                         'size': 10
                         }              # Label text font settings for graphs
        
        self.cbarFont = {'family': 'sans-serif',
                         'color': 'black',
                         'weight': 'normal',
                         'size': 12
                         }
    
    def addEdge(self, u,v):
        """Function to add dan edge to undirected graph"""
        self.graph[u].append(v)
        self.graph[v].append(u)
        #print(self.graph)

    def DFSUtil(self, v, visited, vDegree, k):
        """
        A recursive function to call DFS starting from v. It
        returns True if vDegree of v after processing is less
        than k, else False. It also updates vDegree of adjacent
        vertex if vDegree of v is less than k. If vDegree of v
        becomes less than k, then it reduces vDegree of v also.
        """

        # Mart the current node as visited
        visited[v] = True

        # Recur for all the vertices adjacent to this vertext
        for i in self.graph[v]:
            # If vDegree of v < k, then vDegree of adjacent v reduced
            if vDegree[v] < k:
                vDegree[i] = vDegree[i] - 1
            
            # If adjacent is not processes, process it
            if visited[i] == False:
                if (self.DFSUtil(i, visited, vDegree, k)):
                    vDegree[v] -= 1
        #Return True if vDegree of v is less than k
        return vDegree[v] < k
    
    def printKCores(self, k):
        """Print k-cores of an undirected graph"""
        
        # Initialization: mark all v as not visited, and 
        # store vDegrees of all vertices
        visited = [False]*self.V    
        vDegree = [0]*self.V        
        for i in self.graph:
            vDegree[i] = len(set(self.graph[i]))
        
        # Chose any vertex as starting vertext
        self.DFSUtil(0, visited, vDegree, k) 

        # DFS traversal to update vDegrees of all vertices
        # in the case they are unconnected.
        for i in range(self.V):
            if visited[i] == False:
                self.DFSUtil(i, vDegree, visited, k)

        # Printing K-Cores
        print("\n K-Cores: ")
        for v in range(self.V):
            if vDegree[v] >= k:
                print(str("\n [ ") + str(v) + str(" ]"))
                for i in self.graph[v]:
                    if vDegree[i] >= k:
                        print "->" + str(i)
    
    
    def genAdjMatrix(self, k, fname, opt='default'):
        """Generates adjacency matrix for k-core subgraph"""
        connectivity = defaultdict(list)
        visited = [False]*self.V
        vDegree = [0]*self.V
        for i in self.graph:
            vDegree[i] = len(set(self.graph[i]))
        print(vDegree)
        self.DFSUtil(0, visited, vDegree, k)
        
        for v in range(self.V):
            if vDegree[v] >= k:
                for i in self.graph[v]:
                    if vDegree[i] >= k:
                        connectivity[v].append(i)
        
        if opt == 'default':
            print("NUM COLRS")
            print len(connectivity.keys())
            print connectivity
            return connectivity.keys()
        
        N = int(self.V)
        proto_mat = np.zeros((N, N))
        
        for i in range(self.V):
            for j, val in enumerate(connectivity[i]):
                proto_mat[i, val] = int(1)
        
        if opt == 'save':
            pref = fname.strip('.csv')
            adj_matrix = DataFrame(proto_mat).astype(int)
            adj_matrix.to_csv('{}_{}-core.csv'.format(pref, k), header=False, sep=',', index=False)
        adj_matrix = np.array(proto_mat)
        return adj_matrix


    def genXY(self, graph_pos, shells):
        """Generate x, y coordinates for each node in graph"""
        x, y, xy = [], [], []
        for sub_shell in shells:
            tempx, tempy, tempxy = [], [], []
            for i in range(len(sub_shell)):
                tempx.append(graph_pos[sub_shell[i]][0])
                tempy.append(graph_pos[sub_shell[i]][1])
                tempxy.append((graph_pos[sub_shell[i]][0], graph_pos[sub_shell[i]][1]))
            x.append(tempx)
            y.append(tempy)
            xy.append(tempxy)
        return x, y


    def core_processing(self, shells):
        for i in range(len(shells)-1):
            shells[i+1] = [x for x in shells[i+1] if x not in shells[i]]
        return shells
                
    

    def core_decomposition(self, gr, fname):
        node_list = gr.nodes()
        degrees = list(set(dict(gr.degree(node_list)).values()))
        print 
        core_nodes = []
        core_nodes = {}
        for deg in list(sorted(degrees, reverse=True)):
            temp = self.genAdjMatrix(deg, fname)
            core_nodes[deg] = temp
        print("CORE STUFF")
        core_nodes = {k: v for (k, v) in core_nodes.items() if v != []} 
        print core_nodes
        shell_data = {}
        degs = sorted(core_nodes.keys(), reverse=True)
        for i in range(len(degs)):
            if i == 0:
                shell_data[degs[0]] = core_nodes[degs[i]]
            elif i == 1:
                temp = list(set(core_nodes[degs[0]])\
                        .symmetric_difference(set(core_nodes[degs[i]])))
                shell_data[degs[i]] = temp
            else:
                 temp = list(set(core_nodes[degs[i-1]])\
                         .symmetric_difference(set(core_nodes[degs[i]])))
                 shell_data[degs[i]] = temp
        
        shells = []
        for shell in sorted(shell_data.keys(), reverse=True):
            shells.append(shell_data[shell])
        print shells
        return shell_data, shells


    def pointsInCircum(self, r, n):
        coords = []
        for d in range(0,n+1):
            x = cos(2*pi/n*d)*r
            y = sin(2*pi/n*d)*r
            coords.append(np.array([x, y]))
        return coords

    
    def gen_coords(self, shells, graph_pos, rads):
        print shells
        shell_rads = {}
        
        for i in range(len(shells)):
            if i == 0:
                shell_rads[i] = rads[i]*0.8
            else:
                shell_rads[i] = (rads[i] + rads[i-1])/2.0
        
        for i, shell in enumerate(shells):
            if len(shell) < 3:
                x_r = np.linspace(-shell_rads[i], shell_rads[i], len(shell))
                for j in range(int(len(shell))):
                    y_i = -1.0 * sqrt((shell_rads[i]**2) - (x_r[j])**2)
                    graph_pos[shell[j]] = np.array([x_r[j], y_i])
            
            if len(shell) >= 3:
                coords = self.pointsInCircum(shell_rads[i], len(shell))
                for j in range(len(shell)):
                    graph_pos[shell[j]] = coords[j]
        return graph_pos


    def gen_rads(self, shells):
        if len(shells) <= 6:
            mini = 0.8
        else:
            mini = 0.6
        if len(shells) <= 2:
            maxi = 1.0
        else:
            maxi = (0.35 * len(shells)-1) + mini
        rads = np.linspace(mini, maxi, len(shells))
        return rads, mini, maxi


    #TODO: Need to be able to color nodes based on ecc and size nodes based on bc
    def draw_graph(self, graph, labels=None, graph_layout='shell',
                   node_size=1600, node_colors='blue', node_alpha=0.5,
                   node_text_size=8, edge_color='grey', edge_alpha=0.2,
                   edge_thickness=1, edge_test_pos=0.3, label_edges=False,
                   edge_labels=None, show=False, customize=False,
                   text_font='sans-serif', community_analysis=False):
        
        #TODO: need to add in functionality if node_colors is a list ::
        #to color every node differently. Also need to modify for community
        #graphing.

        #Initiate graph construction
        gr = nx.Graph()
        rows, cols = np.where(graph == 1)
        graph = zip(rows.tolist(), cols.tolist())
        gr.add_edges_from(graph)
        
        # These are different layouts for the network you may try
        # shell seems to work the best.if not specified Networkx
        # defaults to 'spring' layout
        if graph_layout == 'spring':
            graph_pos = nx.spring_layout(gr)
        elif graph_layout == 'spectral':
            graph_pos = nx.spectral_layout(gr)
        elif graph_layout == 'bipartite':
            graph_pos = nx.bipartite(gr)
        elif graph_layout == 'kk':
            graph_pos = nx.kamada_kawai_layout(gr)
        elif graph_layout == 'random':
            graph_pos = nx.random_layout(gr)
        elif graph_layout == 'cores':
            shell_data, shells = self.core_decomposition(gr, self.file_)
            rads, mini, maxi = self.gen_rads(shells)
            pref = 'core-decomp'
            graph_pos = nx.shell_layout(gr, shells)
            X, Y = self.genXY(graph_pos, shells)
            graph_pos = self.gen_coords(shells, graph_pos, rads)
        
        #elif community_analysis == True:
        #    graph_pos = nx.
        else:
            graph_pos = nx.shell_layout(gr)

        # Draw graph
        # If customize: color nodes according to ecc and size them by bc
        ax = plt.gca()
        if customize:
            if len(rads) > 2:
                node_text_size=8
            ne, nv, bcs, ecs, unique_degs = self.graph_properties(gr)
            bcs = {key: val+max(bcs.values()) for (key, val) in zip(bcs.keys(), bcs.values())}
            bcs = {key: val*250 for (key, val) in zip(bcs.keys(),
                            [x/max(bcs.values()) for x in bcs.values()])}
            
            sizes = defaultdict(list)
            distinct_bcs = list(set(bcs.values()))
            for val in distinct_bcs:
                for node, size in bcs.items():
                    if size == val:
                        sizes[size].append(node)
            node_sizes = [int(x/len(shells)) if x == min(bcs.values()) else x for x in bcs.values()]
            clrs = {}
            print node_sizes
            print "ECC"
            print graph_pos
            for key, val in ecs.items():
                clrs[key] = str(self.eccColors[int(val)-1])

            nx.draw_networkx_nodes(gr, graph_pos, node_size=node_sizes,
                            node_color=clrs.values(), alpha=node_alpha)
        else:
            nx.draw_networkx_nodes(gr, graph_pos, node_size=node_size,
                               alpha=node_alpha, node_color=node_colors)
        
        nx.draw_networkx_edges(gr, graph_pos, width=edge_thickness,
                               alpha=edge_alpha, color=edge_color)
        
        
        # Generate default node labels (numbered from 1 -> len(graph))
        if labels is None:
            label_pos = {}
            node_list = gr.nodes()
            ###ADDED SORTED IN LINE BELOW CHECK US RESULTS!!
            labels = list(sorted(set([x+1 for x in rows.tolist()])))
            node_labels = dict(zip(node_list, labels)) 
            print node_list
            print labels
            print("Node labels")
            print node_labels
            for k, v in graph_pos.items():
                label_pos[k] = (v[0] +(0.01*v[0]), v[1] + (0.1*v[1]))
            nx.draw_networkx_labels(gr, graph_pos, labels=node_labels, 
                            font_size=node_text_size, font_family=text_font)

        # If set true: label edges with same labels as nodes
        if label_edges and type(edge_labels) is not list:
            edge_labels = dict(zip(gr, labels))
            nx.draw_networkx_edge_labels(gr, graph_pos, edge_labels=edge_labels,
                                         label_pos=edge_text_pos)
        
        # Edges labeled by values provided in list: good for labelling edges
        # by weights: needs to be dict need to check how this works
        if label_edges and type(edge_labels) is list:
            edge_labels = dict(zip(graph, label_edges))
            nx.draw_networkx_edge_labels(gr, graph_pos, edge_labels=edge_labels,
                                         label_pos=edge_text_pos)
        if graph_layout == 'cores':
            for j in range(len(rads)):
                if len(shells) == 1:
                    lab = "k = {}".format(shell_data.keys()[0])
                else:
                    lab="default"
                self.encircle(ax, rads[j], j, label=lab)
            
            if len(shells) != 1:
                self.shell_color_bar(ax, shell_data, maxi)
            cmap = plt.cm.jet
            cmaplist=self.eccColors
            cmap = cmap.from_list('custom_map', cmaplist, len(cmaplist))
            bounds = list(range(1, 7))# In future set to (1, max(ecc)+2)
            norm = mpl.colors.BoundaryNorm(bounds, len(cmaplist))#cmap.N)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                         spacing='uniform', format='%1i')#, ticks=[1, 2, 3, 4, 5],
                                         #boundaries=[1, 2, 3, 4, 5], format='%1i')
            cb.ax.set_ylabel("Eccentricity", fontdict=self.cbarFont)  
            tick_locs = (np.arange(len(bounds)) + 0.5) #* (len(bounds)-1)/len(bounds)
            cb.set_ticks(tick_locs)
            ax.set_xlim((-maxi-(0.4*maxi)), (maxi+0.25))
            ax.set_ylim((-maxi-0.25), (maxi+0.25))
        else:
            pref='other'
       
        ax.set_axis_off()
        if show:
            plt.show()
        pname = '_'.join([pref, self.file_.strip('.csv')])
        #plt.savefig("{}.png".format(pname))
        #plt.close()

    def shell_color_bar(self, ax, shell_data, maxi):
        shells = sorted(shell_data.keys())
        nShells = len(shells)
        cmaplist=[]
        for i, shell in enumerate(shells):
            cmaplist.append(self.shlColors[i])
            #if nShells == 1:
                #cmaplist.append(self.shlColors[i+1])
                #cmaplist.append(self.shlColors[i+2])
        cmap = plt.cm.jet
        cmaplist=cmaplist[::-1]
        cmap = cmap.from_list('shell_map', cmaplist, len(cmaplist))
        bounds = list(range(1, nShells+2))
        
        norm = mpl.colors.BoundaryNorm(bounds, len(cmaplist))
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="5%", pad=0.05)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, ticks=shells,
                                       spacing='uniform', format='%1i')
        cb.ax.set_ylabel("K-shell", fontdict=self.cbarFont)
        cb.ax.yaxis.set_label_coords(-1.2, 0.5)
        tick_locs = (np.arange(len(bounds))+0.5)
        cb.set_ticks(tick_locs)
        cb.set_ticklabels(np.array(shells))


    def encircle(self, ax, radius, j, label, center=(0,0)):
        """Encircle and label region of given k-core. Works best with shell"""
        ec = self.shlColors[j]
        circle = plt.Circle(center, radius=radius, ec=ec, fc='none',
                            linewidth=1.5, linestyle='--')
        ax.add_patch(circle)
        if label != "default":
            plt.text(-0.14, radius+0.035, label, fontdict=self.textFont)
        
    
    def get_rads(self, x, y):
        #### REMOVE FUNCTION IT IS NOT USED ####
        rads = []
        for i in range(len(x)):
            if max(x[i]) < 0.0:
                rads.append(max(y[i])+0.08)
            else:
                rads.append(max(x[i])+0.08)
        print rads
        return rads


    def poly_encircle(x, y, ax=None, **kw):
        """Encircle regions in graphs. Not built to work with 3D graphs"""
        if not ax: 
            ax=plt.gca()
        p = np.c_[x, y]
        hull = ConvexHull(p)
        poly = plt.Polygon(p[hull.vertices,:], **kw)
        ax.add_path(poly)


    def draw_graph3d(self, graph, graph_colormap='winter', bgcolor=(1, 1, 1),
                     node_size=0.03, edge_color=(0.8, 0.8, 0.8), edge_seize=0.002,
                     text_size=0.008, text_color=(0, 0, 0)):
        
        # Initiate graph construction
        gr = nx.Graph()
        rows, cols = np.where(graph == 1)
        graph = zip(rows.tolist(), cols.tolist())
        gr.add_edges_from(graph)
        
        
        gl = nx.convert_node_labels_to_integers(gr)
        
        graph_pos = nx.spring_layout(gl, dim=3)
        
        # Numpy array of x,y,z positions in sorted node order
        xyz = np.array([graph_pos[v]] for v in sorted(gl))

        # Scalar colors
        scalars = np.array(gl.nodes())+5
        mlab.figure(1, bgcolor=bgcolor)
        mlab.clf()

        pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
                            scalars, scale_factor=node_size,
                            scale_mode='none', colormap=graph_colormap,
                            resolution=20)
        
        for i, (x, y, z) in enumerate(xyz):
            label = mlab.text(x, y, str(i), z=z, width=text_size, name=str(i),
                              color=text_color)
            label.property.shadow = True

        pts.mlab_source.dataset.lines = np.array(gl.edges())
        tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
        mlab.pipeline.surface(tube, color=edge_color)

        # Show 3D graph in interactive window
        mlab.show()


    def create_tangled_hypercube(self, Nnodes):
        """Create a tangled hypercube with Nnodes for fun"""
        
        # Initiate nodes
        nodes = range(Nnodes)

        def make_link(graph, i1, i2):
            graph[i1][i2] = 1
            graph[i2][i1] = 1

        if Nnodes == 1: 
            return {nodes[0]:{}}

        nodes1 = nodes[0:Nnodes/2]
        nodes2 = nodes[Nnodes/2:]
        G1 = create_tangled_hypercube(nodes1)
        G2 = create_tangled_hypercube(nodes2)

        # Merge G1 and G2 into a single graph
        G0 = dict(G1.items() + G2.items())

        # Link G1 and G2
        random.shuffle(nodes1)
        random.shuffle(nodes2)
        for i in range(len(nodes1)):
            make_link(G0, nodes1[i], nodes2[i])
        

        draw_graph3d(G0)
   

    def graph_properties(self, gr):
        ne = gr.number_of_edges()
        nv = gr.number_of_nodes()
        bcs = nx.betweenness_centrality(gr)
        ecs = nx.eccentricity(gr)
        degrees = list(gr.degree())
        unique_degs = set(list(dict(degrees).values()))

        return ne, nv, bcs, ecs, unique_degs

    
class MainProgram:
    
    def __init__(self):
        opts = self.parse_options()

    def parse_options(self):
        """Parse user input options to determine workflow and 
           graph esthetics"""
       
        parser = argparse.ArgumentParser(description='Graph and core analysis tool')
        
        parser.add_argument('-f', '--files', type=str, nargs='+', required=True,
            help="(Required):: Adjacency matrix csv file(s). For multiple use a\
                  space separated sequence.")

        parser.add_argument('-c', '--custom', type=bool, nargs=1, default=False,
            help="(Optional):: Customize 2D graph:: if True will set up graph for\
                  core analysis, color nodes according to their eccentricity, and\
                  size them by their betweenness centrality. (Default=False)")

        parser.add_argument('-nl', '--node_labels', nargs='+', default=None,
            help="(Optional):: Set custom node labels. If not set the labels will\
                  be added such that nodes are labelled by there index in the\
                  adjacency matrix that is supplied. (Default=None)")

        parser.add_argument('-gl', '--graph_layout', type=str, nargs=1, default='shell',
            help="(Optional):: Determines graph drawing method in Networkx.\
                  Methods include: spring, spectral, kk, random, cores, shell.\
                  Set to 'cores' for core analysis. (Default=shell).")

        parser.add_argument('-ns', '--node_size', type=int, nargs=1, default=1600,
            help="(Optional):: Set size of nodes. If 'custom' set to True, the\
                  nodes will be sized according to their betweenness-\
                  centrality. (Default=1600)")

        parser.add_argument('-nc', '--node_colors', type=str, nargs='+', default='blue',
            help="(Optional):: Set color(s) of nodes. Input sequence of colors\
                  with length equal to the number of nodes in graph to color\
                  nodes accordingly. If 'custom' is set to True, nodes will be\
                  colored according to their eccentricity.")

        parser.add_argument('-na', '--node_alpah', type=float, nargs=1, default=0.8,
            help="(Optional):: Set transparency of nodes (0.0 - 1.0). (Default=0.8)")

        parser.add_argument('-nts', '--node_text_size', type=int, nargs=1, default=12,
            help="(Optional):: Set node label fontsize. (Default=12)")

        parser.add_argument('-ec', '--edge_color', type=str, nargs=1, default='grey',
            help="(Optional):: Set color of edges in graph. This option currently\
                  only works for single color. (Default=grey)")

        parser.add_argument('-ea', '--edge_alpha', type=float, nargs=1, default=0.2,
            help="(Optional):: Set transparency of edges. Default setting\
                  achieves best visibility. (Default=0.2)")

        parser.add_argument('-et', '--edge_thickness', type=float, nargs=1, default=1.0,
            help="(Optional):: Set thickness of edges. Default setting acheives\
                  best visibility. (Default=1.0)")

        parser.add_argument('-le', '--label_edges', type=bool, nargs=1, default=False,
            help="(Optional):: Set label edges or not. (Default=False)")

        parser.add_argument('-el', '--edge_labels', type=str, nargs='+', default=False,
            help="(Optional):: if -et True, this argument is used to specify the\
                  the edge labels as a sequence of space separated values. Good\
                  for specificying weights. Values read in as type(list).\
                  (Default=False)")

        parser.add_argument('-s', '--show', type=bool, nargs=1, default=False,
            help="(Optional):: Option to show graph or not. Regardless if\
                  graph is shown or not, the graph will be saved as a png file.\
                  (Default=False)")

        parser.add_argument('-tf', '--text_font', type=str, nargs=1, default='sans-serif',
            help="(Optional):: Select font family. (Default=sans-serif)")

        parser.add_argument('-ca', '--com_analys', type=bool, nargs=1, default=False,
            help="(Optional):: Generate plot depicting distinct communities.\
                 If both -ca and -c are set True, a subplot with both graphs\
                 is created side by side. If one or the other set True a\
                 single graph will be generated. (Default=False)")

        parser.add_argument('-thc', '--tangled', type=bool, nargs=1, default=False,
            help="(Optional):: Generate 3-dimensional graph. This option has not\
                  been full tested and still requires set up. (Default=False)")


        return parser.parse_args()

                    
    def initialize_graph(self, fname):
        """Create graph for analysis from adjacency matrix"""
        adj_mat = DataFrame.from_csv(fname, header=None, index_col=False).values

        G = GraphGen(len(adj_mat), fname)
        for i in range(len(adj_mat)):
            for j, val in enumerate([float(x) for x in adj_mat[i]]):
                if val > 0.0:
                    G.addEdge(int(i), int(j))
        
        return G, adj_mat

    
    def workflow(self):
        """Use user defined options to determine workflow"""
        
        uo = self.parse_options()
        
        if uo.graph_layout[0]=='cores':
            """Perform core-decomposition: -f [files] -gl cores 
               -s True -c True"""
            uo.custom=True
            for fname in uo.files:
                G, graph = self.initialize_graph(fname)
                G.draw_graph(graph, graph_layout=uo.graph_layout[0], show=uo.show[0],
                             customize=uo.custom)
                

if __name__ == '__main__':
    prog = MainProgram()
    prog.workflow()
