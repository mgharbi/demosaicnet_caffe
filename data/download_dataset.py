#!/usr/bin/env python
import os
import platform
import subprocess

def main():
    files = ["datasets.z{:02d}".format(i+1) for i in range(41)]
    files.append("datasets.zip")

    checksums = {
        'datasets.z01': 'd8afdafd00b34269e751e3bbc3324b0a',
        'datasets.z02': '864d674842b528ccf7e6c1da278e57ed',
        'datasets.z03': '4b3b76a2b2bd8931668e27958575c9fc',
        'datasets.z04': '7b2443268e9596ea71ced9540de45a17',
        'datasets.z05': '16b39e42184a41f51669eeb0b6d0f611',
        'datasets.z06': '9faf954cd993b7266116eaf87a135d50',
        'datasets.z07': 'd2d03740c37c2388f4139bb5909d3827',
        'datasets.z08': 'e50ce91f8a572864baddceae45566051',
        'datasets.z09': 'bf83ad30670d96d03b22be9d678f5ffd',
        'datasets.z10': 'a23f4021bcb2bac67d5f3f33b0519cfd',
        'datasets.z11': '51594d190a75f8bcf21d7dafb9bd1da7',
        'datasets.z12': '07eca17e1671aa7e223794899a7a9f08',
        'datasets.z13': 'd4d3c68c985d281a75a09a176bfc28cc',
        'datasets.z14': '6b35903318ff71486b2582fecc9feb95',
        'datasets.z15': 'f896d84edfb64e704302851dc5f02b67',
        'datasets.z16': '0d0dde082d41994a43eda8d3b91bd230',
        'datasets.z17': '288beca9219868278e85c3e0d4a94f7b',
        'datasets.z18': '8932d1e0d4d69cc7bd8291f97c516d62',
        'datasets.z19': 'c8ed240ae4a9bdc71c37a9edee260364',
        'datasets.z20': '381ab481a820e62b1ef194913c511329',
        'datasets.z21': '1eba7f7ae6bc36819af38a0cd45f5ed9',
        'datasets.z22': '9044b3098609c61c7ecf9ae8709634b5',
        'datasets.z23': '0549d3bf164b98f60e2a3f17c6fbc5e6',
        'datasets.z24': '3d4cff2503b06360fb8ed9e52a2c9c5c',
        'datasets.z25': 'b31aabd0cb9c56cb4f87529534908c63',
        'datasets.z26': 'b35b851fa2397eb724f8c5c676d28bf1',
        'datasets.z27': 'df151afae7c54bebc286ddf57a6eea39',
        'datasets.z28': 'fbf889d24bc035e93a4a4b50dc956126',
        'datasets.z29': '7f2f7b39f4e7a9c1f5732152c759e456',
        'datasets.z30': 'a943c817b38818ab4a5ac38cbc790370',
        'datasets.z31': '130b007b09021e9fd48c9175a6bc8088',
        'datasets.z32': 'ce2a09a8dc9e204f8ff16e203e257dab',
        'datasets.z33': 'd58794da4fc24116e0693be09c2804d9',
        'datasets.z34': '0e52378d3e783f99003f2156e3f88bbb',
        'datasets.z35': '23efba745765ab6ad07cb524b72d2b3a',
        'datasets.z36': 'fb95689a11ca49c3596a1f031250d6cb',
        'datasets.z37': '5a3f3b21b9e882790b2a7a3965f05968',
        'datasets.z38': '5f60b08de88884db1cb37a953b0903ad',
        'datasets.z39': '42751d1278aed676d439a79e866ad219',
        'datasets.z40': '892cb82bd8e97092f371370d1ce659a8',
        'datasets.z41': '0df61a460c166c36ad646b42ef106a5c',
        'datasets.zip': '4bcfa692f2f259cea7cc23da46f3f79e',
    }

    url_root = "https://data.csail.mit.edu/graphics/demosaicnet"

    dst = os.path.dirname(os.path.abspath(__file__))

    for f in files:
        fname = os.path.join(dst, f)
        url = os.path.join(url_root, f)
        if os.path.exists(fname):
            print fname, 'already downloaded. Checking md5...',
            if platform.system() == "Linux":
                check = subprocess.Popen(['md5sum', fname], stdout=subprocess.PIPE)
                checksum = subprocess.Popen(['awk', '{ print $1 }'], stdin=check.stdout, stdout=subprocess.PIPE)
                checksum = checksum.communicate()[0].strip()
            elif platform.system() == "Darwin":
                check = subprocess.Popen(['cat', fname], stdout=subprocess.PIPE)
                checksum = subprocess.Popen(['md5'], stdin=check.stdout, stdout=subprocess.PIPE)
                checksum = checksum.communicate()[0].strip()
            else:
                raise Exception("unknown platform %s" % platform.system())

            if checksum == checksums[f]:
                print "MD5 correct, no need to download."
                continue
            else:
                print "MD5 incorrect, re-downloading."
            print checksum, checksums[f]

        cmd = ['curl', url, '-O', fname]
        print "Running", " ".join(cmd)
        ret = subprocess.call(cmd)

    print "Joining zip files"
    ret = subprocess.call(["zip", "-FF", os.path.join(dst, "datasets.zip"), "--out", os.path.join(dst, "joined.zip")], shell=True)
    print ret

    print "Extracting files"
    subprocess.call(["unzip", os.path.join(dst, "joined.zip")], shell=True)

if __name__ == '__main__':
    main()
