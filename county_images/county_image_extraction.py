from mapbox import Static
import mapbox
import pandas as pd
import mercantile
import os
# Import the image, math and os libraries
import PIL
import math 
from PIL import Image
from os import listdir
from os.path import isfile, join
import requests # The requests package allows use to call URLS
import shutil   # shutil will be used to copy the image to the local

def coord_range_breakdown(bbox):
    # Topleft / Bottom right coordinates of the bounding box
    tl = [float(bbox['corner_nw'][0].split(',')[0]), float(bbox['corner_nw'][0].split(',')[1])]
    br = [float(bbox['corner_se'][0].split(',')[0]), float(bbox['corner_se'][0].split(',')[1])]
     # Set the resolution (max at 15)
    tl_tiles = mercantile.tile(tl[1],tl[0],z)
    br_tiles = mercantile.tile(br[1],br[0],z)

    return [tl_tiles.x,br_tiles.x], [tl_tiles.y,br_tiles.y]

def loop_over_tile(x_tile_range, c, z ):
    # Loop over the tile ranges
    for i,x in enumerate(range(x_tile_range[0],x_tile_range[1]+1)):
        for j,y in enumerate(range(y_tile_range[0],y_tile_range[1]+1)):
        # Call the URL to get the image back
        r = requests.get('https://api.mapbox.com/v4/mapbox.terrain-rgb/'+str(z)+'/'+str(x)+'/'+str(y)+'@2x.pngraw?access_token=pk.eyJ1IjoiZWZhd2UiLCJhIjoiY2tjb2QwamVyMGZlajJ5bWtxeDNmbTFkciJ9.IPLWkRMYkSmxoUFlCAMZIg', stream=True)
        # Next we will write the raw content to an image
        with open('./elevation_images/' + str(i) + '.' + str(j) + '.png','wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f) 
        # Do the same for the satellite data
        r =requests.get('https://api.mapbox.com/v4/mapbox.satellite/'+str(z)+'/'+str(x)+'/'+str(y)+'@2x.pngraw?access_token=pk.eyJ1IjoiZWZhd2UiLCJhIjoiY2tjb2QwamVyMGZlajJ5bWtxeDNmbTFkciJ9.IPLWkRMYkSmxoUFlCAMZIg', stream=True)
        with open('./satellite_images/' + str(i) + '.' + str(j) + '.png','wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)


def main():
    os.mkdir('composite_images/')
    os.mkdir('elevation_images/')
    os.mkdir('satellite_images/')
    z = 14
    bbox = pd.read_csv('bounding.txt') 
    x_tile_range, y_tile_range = coord_range_breakdown(bbox)
    loop_over_tile(x_tile_range,y_tile_range, z)
    # Loop over the elevation and satellite image set
    for img_name in ['elevation','satellite']:
        # Make a list of the image names   
        image_files = ['./'+img_name+'_images/' + f for f in listdir('./'+img_name+'_images/')]
            # Open the image set using pillow
        images = [Image.open(x) for x in image_files]
        # Calculate the number of image tiles in each direction
        edge_length_x = x_tile_range[1] - x_tile_range[0]
        edge_length_y = y_tile_range[1] - y_tile_range[0]
        edge_length_x = max(1,edge_length_x)
        edge_length_y = max(1,edge_length_y)
        # Find the final composed image dimensions  
        width, height = images[0].size
        total_width = width*edge_length_x
        total_height = height*edge_length_y
        # Create a new blank image we will fill in
        composite = Image.new('RGB', (total_width, total_height))
        # Loop over the x and y ranges
        y_offset = 0
        for i in range(0,edge_length_x):
            x_offset = 0
            for j in range(0,edge_length_y):
                # Open up the image file and paste it into the composed
                tmp_img = Image.open('./'+img_name+'_images/'+ str(i) + '.' + str(j) + '.png')
                composite.paste(tmp_img, (y_offset,x_offset))
                x_offset += width # Update the width
            y_offset += height # Update the height
        # Save the final image
        composite.save('./composite_images/'+img_name+'.png')

main()
