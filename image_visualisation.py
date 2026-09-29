def scale_bar_image(image, 
                    micron_length, 
                    scale_length,
                    scale_color="white",
                    thickness=4,
                    bottom_margin=0.93,
                    left_margin=0.07):                   
    
    """
    Generates a image with a scale bar at the bottom left side
    
    Parameters:
    - image (numpy.ndarray): The input image (2D numpy array) which will be used to generate the scale bar image.
    - micron_length (float): The physical length (in microns) represented by the scale bar.
    - scale_length (float): The desired length of the scale bar in the generated image.
    - scale_color (str, optional): Color of the scale bar. Default is "white".
    - thickness (int, optional): Thickness of the scale bar. Default is 4 pixels.
    - bottom_margin (float, optional): The position of the scale bar from the bottom as a fraction of the image height. Default is 0.93.
    - left_margin (float, optional): The position of the scale bar from the left as a fraction of the image width. Default is 0.07.
    
    Returns:
    - scale_bar (numpy.ndarray): The image with the overlaid scale bar.
    - colormap (matplotlib.colors.ListedColormap): Colormap used for the scale bar.

    Raises:
    - ValueError: If the input image does not have 2 dimensions.
    
    Example:
    
    #import clesperanto
    import pyclesperanto_prototype as cle
        
    #load image
    image = cle.imread("file_path)
    
    #create scale bar and colormap
    scale_bar, colormap = scale_bar_image(image, micron_length=0.323, scale_length=25)
    
    #show image
    cle.imshow(image, continue_drawing=True)
    
    #overlay scale bar
    cle.imshow(scale_bar, colormap=colormap)
    """
    
    if len(image.shape) != 2:
        raise ValueError("Out of index. 2-dimensional images are required.")
    
    from matplotlib.colors import ListedColormap
    import numpy as np
    
    colormap = ListedColormap(["none", scale_color])
    
    scale_bar = np.full(image.shape, np.nan, dtype=float)
    
    length_pixels = int(scale_length / micron_length)
    
    margin_bottom = int(scale_bar.shape[0] * bottom_margin)
    margin_left = int(scale_bar.shape[1] * left_margin)
    
    scale_bar[margin_bottom:margin_bottom + thickness, 
              margin_left:margin_left + length_pixels] = 1
    
    return scale_bar, colormap