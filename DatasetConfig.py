import pandas as pd
import numpy as np
import Preprocessing as prep
from keras.optimizers import Adam





class Datasets():
    def __init__(self, dsConf):
        self._dsConf = dsConf

        # classification column
        self.pathModels = dsConf.get('pathModels')
        self._pathDataset = dsConf.get('pathDataset')
        self._path = dsConf.get('path')

        self._testpath = dsConf.get('testpath')
        self._pathDatasetNumeric = dsConf.get('pathDatasetNumeric')
        self._pathOutputTrain = self._pathDatasetNumeric + self._path + 'Numeric.csv'
        self._pathTest = dsConf.get('pathTest')
        



    def preprocessing(self):
        if ((self._testpath == 'KDDCUP99') or (self._testpath == 'KDDTest+')):
            train_df = pd.read_csv(self._pathDataset + self._dsConf.get('path') + '.csv')
            test_df = pd.read_csv(self._pathDataset + self._dsConf.get('pathTest') + '.csv')
            self._cls = train_df.columns[-1]

            columns = set(train_df.columns)
            print(len(columns))
            listBinary = [' land', ' logged_in', ' root_shell', ' su_attempted', ' is_host_login',
                          ' is_guest_login']
            listNominal = [' protocol_type', ' service', ' flag']
            listNumerical = set(columns) - set(listNominal) - set(listBinary)
            listNumerical.remove(self._cls)

            print(listNumerical)
            train_df, test_df = prep.ohe(train_df, test_df, listNominal)
            train_df, test_df = prep.scaler(train_df, test_df, listNumerical)
            train_df[" classification."].replace(to_replace=dict(normal=1, R2L=0, Dos=0, Probe=0, U2R=0),
                                                 inplace=True)
            test_df[" classification."].replace(to_replace=dict(normal=1, R2L=0, Dos=0, Probe=0, U2R=0),
                                                inplace=True)

            train_df.to_csv(self._pathDatasetNumeric + 'Train_standard.csv', index=False)
            test_df.to_csv(self._pathDatasetNumeric + 'Test_standard' + self._testpath + '.csv', index=False)


        elif (self._testpath == 'UNSW_NB15'):
            print('Using:' + self._testpath)

            train_df = pd.read_csv(self._pathDataset + self._dsConf.get('path') + '.csv')
            test_df = pd.read_csv(self._pathDataset + self._dsConf.get('pathTest') + '.csv')
            train_df = train_df.drop(['id', 'label'], axis=1)
            test_df = test_df.drop(['id', 'label'], axis=1)
            train_df = train_df.rename(columns={'attack_cat': 'classification'})
            test_df = test_df.rename(columns={'attack_cat': 'classification'})
            self._cls = train_df.columns[-1]
            listBinary = ['is_ftp_login', 'is_sm_ips_ports']
            listNominal = ['proto', 'service', 'state']
            columns = set(train_df.columns)
            listNumerical = set(columns) - set(listNominal) - set(listBinary)
            listNumerical.remove(self._cls)

            print(listNumerical)
            train_df, test_df = prep.ohe(train_df, test_df, listNominal)
            train_df, test_df = prep.scaler(train_df, test_df, listNumerical)
            train_df["classification"].replace(
                to_replace=dict(Normal=1, Reconnaissance=0, DoS=0, Exploits=0, Fuzzers=0, Shellcode=0, Analysis=0,
                                Backdoor=0, Generic=0, Worms=0), inplace=True)
            test_df["classification"].replace(
                to_replace=dict(Normal=1, Reconnaissance=0, DoS=0, Exploits=0, Fuzzers=0, Shellcode=0, Analysis=0,
                                Backdoor=0, Generic=0, Worms=0), inplace=True)

            train_df.to_csv(self._pathDatasetNumeric + 'Train_standard.csv', index=False)
            test_df.to_csv(self._pathDatasetNumeric + 'Test_standard' + self._testpath + '.csv', index=False)

        elif (self._testpath == 'CICIDS2017'):
            print('Using:' + self._testpath)
            train_df = pd.read_csv(self._pathDataset + self._dsConf.get('path') + '.csv')
            train_df = train_df.sample(frac=1).reset_index(drop=True)
            train_df.rename(
                columns={' Flow Packets/s': 'Flow_Packets', 'Flow Bytes/s': 'Flow_Bytes', ' Label': 'classification'},
                inplace=True)
            self._cls ='classification'
            label = ['BENIGN', 'ATTACK']
            print(train_df.columns)
            listCategorical = ['Flow ID, Source IP, Destination IP, Timestamp, External IP']
            train_df['Flow_Bytes'].fillna((0), inplace=True)
            train_df['Flow_Bytes'] = train_df['Flow_Bytes'].astype(float)

            Pack = train_df[train_df.Flow_Packets != 'Infinity']
            Bytes = train_df[train_df.Flow_Bytes != np.inf]
            maxPack = np.max(Pack['Flow_Packets'])
            maxBytes = np.max(Bytes['Flow_Bytes'])
            columns = train_df.columns
            train_df['Flow_Packets'].replace(to_replace=dict(Infinity=maxPack), inplace=True)
            train_df['Flow_Bytes'].replace((np.inf, maxBytes), inplace=True)

            train_df[self._cls].replace(to_replace=dict(BENIGN=1, ATTACK=0), inplace=True)

            listNominal = []  # Gargaro vuoto
            listBinary = [
                'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', 'FIN Flag Count',
                ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
                ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', 'Fwd Avg Bytes/Bulk',
                ' Fwd Avg Packets/Bulk',
                ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
            listNumerical = set(columns) - set(listNominal) - set(listBinary)
            listNumerical.remove(self._cls)

            print(len(listNumerical))
            print(listNumerical)
            tests = list()
            for testset in self._pathTest:
                print(testset)
                test = pd.read_csv(self._pathDataset + testset + ".csv")
                test.rename(
                    columns={' Flow Packets/s': 'Flow_Packets', 'Flow Bytes/s': 'Flow_Bytes',
                             ' Label': 'classification'},
                    inplace=True)
                test['Flow_Packets'].replace(to_replace=dict(Infinity=maxPack), inplace=True)

                test['Flow_Bytes'].replace(to_replace=dict(Infinity=maxBytes), inplace=True)
                test["classification"].replace(to_replace=dict(BENIGN=1, ATTACK=0), inplace=True)

                tests.append(test)

            train_df, test_df = prep.scalerCICIDS(train_df, tests, listNumerical)

            for t, testset in zip(test_df, self._pathTest):
                t.to_csv(self._pathDatasetNumeric + 'Test_standard' + testset + '.csv', index=False)

            train_df.to_csv(self._pathDatasetNumeric + 'Train_standard.csv', index=False)



    def getClassification(self, df):
        self._cls=list()
        self._cls.append([col for col in df.columns if 'classification' in col])
        self._cls.append([col for col in df.columns if 'Classification' in col])
        self._cls.append([])
        print(self._cls)
        _cls=self._cls[0]
        if not _cls:
           _cls=self._cls[1]
       
        self._cls=_cls
        print(self._cls)
        return self._cls[0]




