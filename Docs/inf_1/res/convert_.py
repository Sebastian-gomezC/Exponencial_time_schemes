import os
import sys

import img2pdf
from PIL import Image
 
# storing image path
img_path = sys.argv[1]
(name,ext) = os.path.splitext(img_path)
# storing pdf path
pdf_path = name + ".pdf"
 
# opening image
image = Image.open(img_path)
 
# converting into chunks using img2pdf
pdf_bytes = img2pdf.convert(image.filename)
 
# opening or creating pdf file
file = open(pdf_path, "wb")
 
# writing pdf files with chunks
file.write(pdf_bytes)
 
# closing image file
image.close()
 
# closing pdf file
file.close()
 
# output
print("Successfully made pdf file")
