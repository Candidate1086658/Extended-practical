# Extended-practical
The hBN thickness determination software and animation code

color matching functions.py compares color matching function fits (Fig 1c  in the report)
RGB values fit.py compares AFM characterised and theoretical RGB values (Fig 3a)
color animation V2.py animates the color of hBN flakes of different thickness (Fig 3b)
2D Thickness.py determines given hBN thickness (Fig 4)

You can run all the code conveniently in Visual Studio Code.
If you prefer to run the code in the terminal, run:
(Windows) cd "$env:(folder that the file is in)"
python3 2D Thickness.py

In HBN images.zip, you can find images of hBN flakes of various thicknesses and color
To test the hBN thickness determination software
1. Open 2D Thickness.py in either VS code or the terminal.
2. Select "no camera (demo)".
3. Upload an image of hBN from HBN images.zip
4. Select hBN mode, select "measure mode"
5. Drag squares on your chosen hBN flake and substrate nearby, the best fit thickness will be shown
6. If "individual channel fittings mismatch" is shown, click "individual channel fittings" to further check the possible thicknesses
7. Other functions: You can change "exposure" to change the brightness, stick to the same region of substrate using "hold", and sketch on the image using "draw" mode. You can also take a screenshot (usually used under live camera)

8. 
