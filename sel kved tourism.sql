SELECT A.*, (SELECT DISTINCT LEVEL_3 FROM ETALON.E_KATOTTG B WHERE D_END>SYSDATE AND A.ID_KATOTTG=B.C_KATOTTG) L3 FROM STSU.R21TAXPAY_U A WHERE KVED IN (
select KOD from etalon.e_kved where d_end>sysdate and kod in (
'_55.10',
'_55.20',
'_55.30',
'_55.90',
'_68.20',
'_56.10',
'_56.29',
'_56.30',
'_79.11',
'_79.12',
'_79.90')) AND D_CLOSE IS NULL AND id_KATOTTG IS NOT NULL and d_reg_sa<trunc(sysdate,'YY') AND FACE_MODE=1 ;

SELECT A.*, (SELECT DISTINCT LEVEL_3 FROM ETALON.E_KATOTTG B WHERE D_END>SYSDATE AND A.ID_KATOTTG=B.C_KATOTTG) L3 FROM STSU.R21TAXPAY_U A WHERE KVED IN (
select KOD from etalon.e_kved where d_end>sysdate and kod in
('_47.19', '_47.78',
'_49.10',
'_49.32', '_49.39',
'_50.10', '_50.30',
'_51.10',
'_77.11',
'_90.03', '_90.04', '_91.02', '_91.03',
'_91.04', '_77.21', '_92.00', '_93.11', '_93.13', '_93.19', '_93.21',
'_86.10', '_86.90', '_96.04',
'_64.19', '_65.12', '_66.22',
'_68.10', '_68.32',
'_63.11', '_63.12',
'_80.10', '_80.20',
'_82.30')) AND D_CLOSE IS NULL AND id_KATOTTG IS NOT NULL and d_reg_sa<trunc(sysdate,'YY') --AND FACE_MODE=1;