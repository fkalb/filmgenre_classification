:: SCRIPT 1:
:: This script moves every folder and their corresponding subfolders one level up. 
:: Make sure the filepath in line 6 is correct, i.e. XXX has to be substituted by your parent directories. This should be done in all following scripts!
for /f "delims=" %%a in ('dir /ad /b') do (
for /f "delims=" %%b in ('dir "%%a" /ad /b') do (
move "%%a\%%b" "XXX\filmgenre_classification\0_corpus"
)
)
pause