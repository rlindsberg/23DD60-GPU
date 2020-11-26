# 23DD60-GPU

Instructions for connecting to Tegner:
https://www.pdc.kth.se/support/documents/login/fedora_login.html
https://canvas.kth.se/courses/20917/pages/tutorial-using-gpus-on-tegner?module_item_id=270532

yum install krb5-workstation openssh-clients

kinit --forwardable ruliu@NADA.KTH.SE

ssh -o GSSAPIDelegateCredentials=yes -o GSSAPIKeyExchange=yes \
    -o GSSAPIAuthentication=yes ruliu@tegner.pdc.kth.se
