# Mise en route de la blockchain

Se déplacer à la racine de la blockchain

```
cd hyperledger/fabric-samples/commercial-paper 
```
Lancer le script de déploiement des différentes images docker
```
./network-start.sh
```

Enroller les utilisateurs
-------------------------


Se déplacer dans le dossier associé aux organisation puis lancer le script enrollUSer

```
cd organisations
node enrollUser.js
```

Déployer les contracts
----------------------

Dans l'une des organisations (ici magnetocorp):
```
cd organisations/magnetocorp
source magnetocorp.sh
peer lifecycle chaincode package cp.tar.gz --lang node --path ./contract --label cp_0
peer lifecycle chaincode install cp.tar.gz
export PACKAGE_ID=cp_0:815155aeebb4cb874f53c0397a4e593f61f00bb99b1df471b60b53567e9d8c44  #numéro de package à modifier
peer lifecycle chaincode approveformyorg --orderer localhost:7050 --ordererTLSHostnameOverride orderer.example.com --channelID mychannel --name bookcontract -v 0 --package-id $PACKAGE_ID --sequence 1 --tls --cafile $ORDERER_CA
```
Dans l'autre organisation (ici digibank) :

```
cd organisations/digibank
source digibank.sh
peer lifecycle chaincode package cp.tar.gz --lang node --path ./contract --label cp_0
peer lifecycle chaincode install cp.tar.gz
export PACKAGE_ID=cp_0:815155aeebb4cb874f53c0397a4e593f61f00bb99b1df471b60b53567e9d8c44 #numéro de package à modifier
peer lifecycle chaincode approveformyorg --orderer localhost:7050 --ordererTLSHostnameOverride orderer.example.com --channelID mychannel --name bookcontract -v 0 --package-id $PACKAGE_ID --sequence 1 --tls --cafile $ORDERER_CA
peer lifecycle chaincode commit -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --peerAddresses localhost:7051 --tlsRootCertFiles ${PEER0_ORG1_CA} --peerAddresses localhost:9051 --tlsRootCertFiles ${PEER0_ORG2_CA} --channelID mychannel --name bookcontract -v 0 --sequence 1 --tls --cafile $ORDERER_CA --waitForEvent

```


Identifiants pour la connection sur le site :
---------------------------------------------
- Bibliotheque_1
  - gemlibuser
- Bibliotheque_2
  - ensimaglibuser
- Maison_Edition
  - mdphachette
