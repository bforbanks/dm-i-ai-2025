@echo off
echo Setting up nnUNet environment variables...

set nnUNet_raw=C:\Users\oscar\OneDrive\School\DTU\DM_AI\dm-i-ai-2025\tumor_segmentation\data_nnUNet
set nnUNet_preprocessed=C:\Users\oscar\OneDrive\School\DTU\DM_AI\dm-i-ai-2025\tumor_segmentation\data_nnUNet\preprocessed
set nnUNet_results=C:\Users\oscar\OneDrive\School\DTU\DM_AI\dm-i-ai-2025\tumor_segmentation\data_nnUNet\results

echo Environment variables set:
echo nnUNet_raw: %nnUNet_raw%
echo nnUNet_preprocessed: %nnUNet_preprocessed%
echo nnUNet_results: %nnUNet_results%

echo.
echo Now you can run the stratified split script. 