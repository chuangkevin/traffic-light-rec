#!/bin/zsh
# 一鍵部署到手機(無線 ADB)
# 用法: ./scripts/deploy-phone.sh <device-ip>[:port]
# 前置: 手機已與本機完成無線偵錯配對,且已開啟「USB 偵錯(安全設定)」
set -e
TARGET=${1:?usage: deploy-phone.sh <device-ip>[:port]}
[[ "$TARGET" == *:* ]] || TARGET="$TARGET:5555"
APK="$(dirname "$0")/../app/build/outputs/apk/debug/app-debug.apk"

export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"
adb connect "$TARGET"
adb -s "$TARGET" install -r "$APK"
adb -s "$TARGET" shell am start -n com.example.trafficlight/.MainActivity
echo "deployed & launched on $TARGET"
