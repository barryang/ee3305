###For re-building the workspace if the built executables do not seem to perform as intended 
###(mostly due to careless changes to the package and executables). Requires bd.sh.

#!/bin/bash
echo "Clean building directory"
rm -rf build install log
./bd.sh