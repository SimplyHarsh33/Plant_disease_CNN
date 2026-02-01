@echo off
echo Starting reorganization...
if not exist "Dataset" mkdir "Dataset"

echo Moving folders...
move "Apple" "Dataset\"
move "Bell Pepper" "Dataset\"
move "Cherry" "Dataset\"
move "Corn (Maize)" "Dataset\"
move "Grape" "Dataset\"
move "Peach" "Dataset\"
move "Potato" "Dataset\"
move "Raspberry" "Dataset\"
move "Soybean" "Dataset\"
move "Squash" "Dataset\"
move "Strawberry" "Dataset\"
move "Tomato" "Dataset\"

echo Cleaning up...
if exist "Testing_Database" rd /s /q "Testing_Database"

echo.
echo ===================================================
echo  DONE! Folders moved to 'Dataset'.
echo  You can now run 'python train_model.py'
echo ===================================================
pause
