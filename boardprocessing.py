import pyautogui
import cv2
import pytesseract
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import numpy as np
import os

# loading in the finetuned mobilenet_v2 model that identifies the chess pieces
chess_piece_model = models.mobilenet_v2(pretrained=False)
num_classes = 13
chess_piece_model.classifier[1] = nn.Linear(chess_piece_model.classifier[1].in_features, num_classes)
chess_piece_model.load_state_dict(torch.load('chess_piece_model.pth'))
chess_piece_model.eval()
# screen resolution
screen_width, screen_height = pyautogui.size()

# transform an image for the chess piece identifier model, same as the one in training
piece_id_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match the input size of many pre-trained models
    transforms.Grayscale(num_output_channels=3),  # Convert image to grayscale (3 channels)
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

def is_mostly_ascending(lst):
    ascending_count = 0
    descending_count = 0
    
    # Iterate through the list and compare consecutive elements
    for i in range(1, len(lst)):
        if lst[i] > lst[i - 1]:
            ascending_count += 1
        elif lst[i] < lst[i - 1]:
            descending_count += 1
    
    # Compare the number of ascending and descending pairs
    return ascending_count > descending_count

class BoardView:
    def __init__(self):
        self.board =  [["." for i in range(8)] for j in range(8)]
        # holds the coordinations for each square to help us find stuff easier (x, y)
        self.board_coords = [[(0, 0) for i in range(8)] for j in range(8)]
        self.orientation = None
        self.fen = None
        self.scale = None
        self.bestfit = None
        self.bestfitloc = None
        self.bestfitbottomright = None

    # turning a screenshot of the user's screen to see the current board position
    def process_board(self):
        print("here")

        screenshot = pyautogui.screenshot()
        screenshot.save("full_screen.png")

        # turning the screenshot into greyscale to do template matching
        img_rgb = cv2.imread("full_screen.png")
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 

        # getting image resolution to scale coordinates later cause screenshot resolution can be different from screen
        img_height, img_width = img_gray.shape[:2]
        scale_x = screen_width / img_width
        scale_y = screen_height / img_height

        # read in the template, which is just the blank chessboard
        template = cv2.imread('template.jpg')
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) 

        template_w = cv2.imread('template2.jpg')
        template_w = cv2.cvtColor(template_w, cv2.COLOR_BGR2GRAY) 

        scale_factor_min = 0.25  # Scale down by 50%
        scale_factor_max = 2.5  # Scale up by 250%
        num_scales = 50         # Number of scales to try in this range

        # to keep track of the scale, loc, and value of out best match
        best_match_loc = None
        best_scale = 1.0
        best_max_val = float("-inf")
        threshold = 0.4
        moved = True
        # first check to see if user moved the chessboard or resized it at all
        if self.scale:
            resized_template = cv2.resize(template, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

            # this incase the user is playing the other side like white and not black pieces 
            resized_template_w = cv2.resize(template_w, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            # perform template matching at this scale
            res = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            res_w = cv2.matchTemplate(img_gray, resized_template_w, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
            (_, maxVal_w, _, maxLoc_w) = cv2.minMaxLoc(res_w)
            # fit is about the same is the last time
            best_val = max(maxVal, maxVal_w)
            best_match_loc = maxLoc if maxVal >= maxVal_w else maxLoc_w
            h, w = resized_template.shape[:2]
            bottom_right_x = best_match_loc[0] + int(w)
            bottom_right_y = best_match_loc[1] + int(h)
            template_bottom_right = (bottom_right_x, bottom_right_y)
            if ((abs(best_val - self.bestfit) <= 0.01 and best_match_loc == self.bestfitloc) or 
                (abs(self.bestfitloc[0] - best_match_loc[0]) <= 2 and 
                abs(self.bestfitloc[1] - best_match_loc[1]) <= 2 and 
                abs(self.bestfitbottomright[0] - template_bottom_right[0]) <= 2 and
                abs(self.bestfitbottomright[1] - template_bottom_right[1]) <= 2)):
                moved = False

        # first time trying to find the board or the user has resized or moved the board location
        if not self.scale or moved: 
            print('refinding the board')
            # loop through the multilpe scales and perform template matching at each scale
            for scale in np.linspace(scale_factor_min, scale_factor_max, num_scales):
                resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                # this incase the user is playing the other side like white and not black pieces 
                resized_template_w = cv2.resize(template_w, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                # perform template matching at this scale
                res = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                res_w = cv2.matchTemplate(img_gray, resized_template_w, cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
                (_, maxVal_w, _, maxLoc_w) = cv2.minMaxLoc(res_w)
                # this match is better than the previous best, so we should save it
                if max(maxVal, maxVal_w) > threshold and max(maxVal, maxVal_w) > best_max_val:
                    if maxVal >= maxVal_w:
                        best_max_val = maxVal
                        best_match_loc = maxLoc
                        self.bestfit  = maxVal
                        self.bestfitloc = maxLoc
                    else:
                        best_max_val = maxVal_w
                        best_match_loc = maxLoc_w
                        self.bestfit = maxVal_w
                        self.bestfitloc = maxLoc_w
                    self.scale = scale
        h, w = template.shape[:2]
        if best_match_loc:
            top_left_x = int(best_match_loc[0])
            top_left_y = int(best_match_loc[1])
            # these should be about the same actually
            width_of_square = int((w * self.scale) // 8)
            height_of_square = int((h * self.scale) // 8)

            bottom_right_x = top_left_x + int(w * self.scale)
            bottom_right_y = top_left_y + int(h * self.scale)
            self.bestfitbottomright = (bottom_right_x, bottom_right_y)
            if not self.orientation:
                # this part is to determine the board orientation
                # just getting the bottom left corner strip to read and decide if user is black or white
                bottom_left_x = top_left_x
                bottom_left_y = int((top_left_y) + (7 * height_of_square))
                # y before x here because it rows (vertical) then columns (horizontal)
                try: 
                    bottom_left_square = img_gray[top_left_y:bottom_left_y+height_of_square, bottom_left_x:bottom_left_x + (width_of_square // 4)]
                    _, bottom_left_square_bin = cv2.threshold(bottom_left_square, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    bottom_left_square_resized = cv2.resize(bottom_left_square_bin, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    mean_intensity = np.mean(bottom_left_square_resized)

                    # Invert colors if the image is mostly dark
                    if mean_intensity < 127:  # Adjust threshold if needed
                        bottom_left_square_resized = cv2.bitwise_not(bottom_left_square_resized)
                    #cv2.imwrite("bottomleft.png", bottom_left_square_resized)
                    pytesseract.pytesseract.tesseract_cmd=os.path.normpath(os.getenv("TESSERACT_PATH"))
                    text = pytesseract.image_to_string(bottom_left_square_resized,config='--psm 6 -c tessedit_char_whitelist=0123456789')
                    if len(text) == 0:
                        print("can't determine color")
                    else:
                        ranks = text.split()
                        if is_mostly_ascending(ranks):
                            self.orientation = "black"
                        else:
                            self.orientation = "white"
                except:
                    print("Error determining color user is playing")

            # this part determines the pieces
            # loop through the board image and then predict in each square. store reuslt in board
            for row in range(8):
                for col in range(8):
                    try:
                        square = img_gray[top_left_y + (row * height_of_square) : top_left_y + (row * height_of_square) + height_of_square, 
                                        top_left_x + (col * width_of_square) : top_left_x + (col * width_of_square) + width_of_square]
                        center_y = top_left_y + (row * height_of_square) + (height_of_square // 2)
                        center_x = top_left_x + (col * width_of_square) + (width_of_square // 2)
                        self.board_coords[row][col] = (int(center_x * scale_x), int(center_y * scale_y))

                        image_tensor = piece_id_transform(square)
                        # add a batch dimension
                        image_tensor = image_tensor.unsqueeze(0)
                        # now use the piece id model to figure out what it is
                        with torch.no_grad():
                            outputs = chess_piece_model(image_tensor)

                        _, predicted_class = torch.max(outputs, 1)
                        classes = ['b', 'k', 'n', 'p', 'q', 'r', '.', 'B', 'K', 'N', 'P', 'Q', 'R']
                        piece_prediction = classes[predicted_class.item()]
                        self.board[row][col] = piece_prediction
                    except:
                        print("error predicting piece at a square")

            # turn the board into FEN
            fen_list = []
            for row in range(8):
                empty_count = 0
                row_string = ""
                for col in range(8):
                    if self.board[row][col] == ".":
                        empty_count += 1
                    else:
                        if empty_count != 0:
                            row_string += str(empty_count)
                        row_string += self.board[row][col]
                        empty_count = 0
                if empty_count != 0:
                    row_string += str(empty_count)
                fen_list.append(row_string)
            if self.orientation == "black":
                fen_list = reversed(list(map(lambda x: x[::-1], fen_list)))
                self.board = [row[::-1] for row in self.board][::-1]
                self.board_coords =[row[::-1] for row in self.board_coords][::-1]
            fen = "/".join(fen_list)
            self.fen = fen

