:: SCRIPT 4:
:: Moves all files on level two to level one. Has to be executed in the same folder as level one!
For /R %%G in (*.gz) do move "%%G" "%%~PG.."

