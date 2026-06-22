@echo off
chcp 65001  nul

echo =====================================
echo Activate Conda Environment
echo =====================================

call conda activate newsrec

echo.
echo =====================================
echo Move to Project Directory
echo =====================================

cd d Dhanhwa(26.03)main

echo.
echo =====================================
echo 1. NAML + ATT
echo =====================================
python main.py --report_encoder=NAML --user_encoder=ATT --dataset=May --batch_size=8 --world_size=1 --epoch_test

echo.
echo =====================================
echo 2. NAML_noTitle + ATT
echo =====================================
python main.py --report_encoder=NAML_noTitle --user_encoder=ATT --dataset=May --batch_size=8 --world_size=1 --epoch_test

echo.
echo =====================================
echo 3. NAML_noTime + ATT
echo =====================================
python main.py --report_encoder=NAML_noTime --user_encoder=ATT --dataset=May --batch_size=8 --world_size=1 --epoch_test

echo.
echo =====================================
echo 4. NAML_noBody + ATT
echo =====================================
python main.py --report_encoder=NAML_noBody --user_encoder=ATT --dataset=May --batch_size=8 --world_size=1 --epoch_test

echo.
echo =====================================
echo 5. NAML_onlyBody + ATT
echo =====================================
python main.py --report_encoder=NAML_onlyBody --user_encoder=ATT --dataset=May --batch_size=8 --world_size=1 --epoch_test

echo.
echo =====================================
echo 6. NAML + ATT_noPosition
echo =====================================
python main.py --report_encoder=NAML --user_encoder=ATT_noPosition --dataset=May --batch_size=8 --world_size=1 --epoch_test

echo.
echo =====================================
echo 7. NAML_noCategory + ATT
echo =====================================
python main.py --report_encoder=NAML_noCategory --user_encoder=ATT --dataset=May --batch_size=8 --world_size=1 --epoch_test

echo.
echo =====================================
echo 8. CROWN + CROWN
echo =====================================
python main.py --report_encoder=CROWN --user_encoder=CROWN --dataset=May --batch_size=8 --world_size=1 --epoch_test

echo.
echo =====================================
echo All experiments completed.
echo =====================================

pause