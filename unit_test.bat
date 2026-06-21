@echo off
chcp 65001 > nul

echo =====================================
echo Activate Conda Environment
echo =====================================

call conda activate newsrec

echo.
echo =====================================
echo Move to Project Directory
echo =====================================

cd /d "D:\hanhwa(26.03)\main"

echo.
echo =====================================
echo 1. NAML + ATT
echo =====================================
python main.py --report_encoder=NAML --unit_encoder=ATT --dataset=unit --batch_size=2 --world_size=1 --epoch_test

echo.
echo =====================================
echo 2. NAML_noTitle + ATT
echo =====================================
python main.py --report_encoder=NAML_noTitle --unit_encoder=ATT --dataset=unit --batch_size=2 --world_size=1 --epoch_test

echo.
echo =====================================
echo 3. NAML_noTime + ATT
echo =====================================
python main.py --report_encoder=NAML_noTime --unit_encoder=ATT --dataset=unit --batch_size=2 --world_size=1 --epoch_test

echo.
echo =====================================
echo 4. NAML_noBody + ATT
echo =====================================
python main.py --report_encoder=NAML_noBody --unit_encoder=ATT --dataset=unit --batch_size=2 --world_size=1 --epoch_test

echo.
echo =====================================
echo 5. NAML_onlyBody + ATT
echo =====================================
python main.py --report_encoder=NAML_onlyBody --unit_encoder=ATT --dataset=unit --batch_size=2 --world_size=1 --epoch_test

echo.
echo =====================================
echo 6. NAML_noCategory + ATT
echo =====================================
python main.py --report_encoder=NAML_noCategory --unit_encoder=ATT --dataset=unit --batch_size=2 --world_size=1 --epoch_test

echo.
echo =====================================
echo 7. NAML + ATT_noName
echo =====================================
python main.py --report_encoder=NAML --unit_encoder=ATT_noName --dataset=unit --batch_size=2 --world_size=1 --epoch_test

echo.
echo =====================================
echo 8. NAML + ATT_noType
echo =====================================
python main.py --report_encoder=NAML --unit_encoder=ATT_noType --dataset=unit --batch_size=2 --world_size=1 --epoch_test

echo.
echo =====================================
echo 9. NAML + ATT_noNameType
echo =====================================
python main.py --report_encoder=NAML --unit_encoder=ATT_noNameType --dataset=unit --batch_size=2 --world_size=1 --epoch_test

echo.
echo =====================================
echo 10. CROWN + CROWN
echo =====================================
python main.py --report_encoder=CROWN --unit_encoder=CROWN --dataset=unit --batch_size=2 --world_size=1 --epoch_test

echo.
echo =====================================
echo All experiments completed.
echo =====================================

pause