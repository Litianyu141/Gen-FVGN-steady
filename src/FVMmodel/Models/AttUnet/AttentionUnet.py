import torch
from torch import nn
from torch_scatter import scatter
from unet_parts import *
from torch_geometric.nn import radius

class Attention_UNet(nn.Module):
	#inspired by UNet taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
	
	def __init__(self, hidden_size=64, bilinear=True):
		super(Attention_UNet, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(hidden_size, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)

		# OutConv
		self.out_act_ln = nn.LayerNorm(hidden_size)

	def radius_interpolt(self, graph_node,node_embedding):
     
		mesh_pos = graph_node.pos
		grids = graph_node.grids
		B,H,W,C= grids.shape
  
		r = torch.dist(grids[0,0,0],grids[0,1,1])
  
		unstructured_grids = grids.reshape(B * H * W, C)

		batch_y = torch.arange(0,B,1,device=graph_node.x.device).view(-1,1).repeat(1,H*W).reshape(-1)

		'''assign interplot position'''
		assign_index = radius(x=mesh_pos,
							  y=unstructured_grids,
							  r=r,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
							  batch_x=graph_node.batch,
							  batch_y=batch_y
                       		 )

		pixel = scatter(
      		src=node_embedding[assign_index[1]],
			index=assign_index[0],
			dim=0,
			reduce="add",
   		)

		pixel = torch.stack([pixel[i==batch_y].reshape(H,W,self.hidden_size) for i in range(B)],dim=0)

		return pixel.permute(0,3,1,2)

	def forward(self,graph_node,node_embedding):
		
		pixel = self.radius_interpolt(graph_node,node_embedding)
  
		x1 = self.inc(pixel)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		
		'''>>> grid sample to point cloud (graph conv) >>>''' 
		x = x.permute(0,2,3,1).permute(0,3,1,2)
		rt_x = []
		for i in range(graph_node.num_graphs):
			mask = i == graph_node.batch
			cur_query = graph_node.query[mask]
			cur_query = cur_query.reshape(1,-1,1,2)
			cur_x = (
				torch.nn.functional.grid_sample(
					x[i : i + 1], cur_query, align_corners=False
				)
				.squeeze()
				.mT
			)

			rt_x.append(cur_x)

		rt_x = torch.cat(rt_x, dim=0)
  
		'''<<< grid sample to point cloud (graph conv) <<<''' 
		# B,C,H,W = final_x.shape
		# batch_x = torch.arange(0,B,1,device=final_x.device).view(-1,1).repeat(1,H*W).reshape(-1)
  
		# rt_x = knn_interpolate(
      	# 				x=final_x.reshape(-1,C), 
        #                  pos_x=graph_node.grids.reshape(-1,2),
        #                  pos_y=graph_node.pos, 
        #                  batch_x=batch_x,
        #                  batch_y=graph_node.batch,
        #                  k=5)

  
		return self.out_act_ln(rt_x)

