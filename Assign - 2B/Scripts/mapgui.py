import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("COS30019 SCATS")
        self.G = nx.Graph()
        self.pos = {}
        self.node_colors = []
        self.edge_colors = []

        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(self.frame)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(self.control_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT)
        self.status_label = ttk.Label(self.control_frame, text="No data")
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.selected_node = None
        self.selected_edge = None

    def load_data(self):
        try:
            locations_df = pd.read_csv("scats_locations.csv")
            
            locations_df = locations_df[locations_df['Latitude'] != 0.0]
            locations_df = locations_df.groupby('SCATS_ID').agg({
                'Latitude': 'mean',
                'Longitude': 'mean',
                'Location': lambda x: x.iloc[0]
            }).reset_index()

            distances_df = pd.read_csv("distance_matrix.csv")
            
            distances_df = distances_df[distances_df['Distance_km'] < 100]
            
            distances_df['pair'] = distances_df.apply(
                lambda row: tuple(sorted([row['From_SCATS'], row['To_SCATS']])),
                axis=1
            )
            distances_df = distances_df.groupby('pair').agg({
                'Distance_km': 'mean',
                'From_SCATS': 'first',
                'To_SCATS': 'first'
            }).reset_index()

            self.G.clear()
            self.G.add_nodes_from(locations_df['SCATS_ID'])
            
            for _, row in distances_df.iterrows():
                if row['From_SCATS'] != row['To_SCATS']:
                    self.G.add_edge(
                        row['From_SCATS'],
                        row['To_SCATS'],
                        weight=row['Distance_km']
                    )

            self.pos = {
                row['SCATS_ID']: (row['Longitude'], row['Latitude'])
                for _, row in locations_df.iterrows()
            }

            self.node_colors = ['blue'] * len(self.G.nodes)
            self.edge_colors = ['black'] * len(self.G.edges)

            self.status_label.config(text="Data loaded")
            self.draw_graph()

        except Exception as e:
            self.status_label.config(text=f"Data error: {str(e)}")

    def draw_graph(self):
        self.ax.clear()
        # self.ax.set_facecolor('green')
        nx.draw(
            self.G,
            self.pos,
            ax=self.ax,
            node_color=self.node_colors,
            edge_color=self.edge_colors,
            node_size=50,
            with_labels=True,
            font_size=8
        )
        self.ax.set_title("COS30019 Network")
        self.canvas.draw()

    def on_hover(self, event):
        if event.inaxes != self.ax:
            return
        self.draw_graph()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        self.draw_graph()

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()