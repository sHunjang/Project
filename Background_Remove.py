from rembg import remove
from PIL import Image



# 이미지 배경 제거
input = Image.open('/Users/seunghunjang/Desktop/Project/EX_IMG/ex5.PNG') # load image
output = remove(input) # remove background
output.save('Test_3.PNG') # save image