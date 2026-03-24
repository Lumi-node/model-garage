# VHS tape for recording Model Garage demo GIF
# Install VHS: https://github.com/charmbracelet/vhs
# Run: vhs assets/demo.tape

Output assets/demo.gif

Set FontSize 14
Set Width 900
Set Height 600
Set Theme "Catppuccin Mocha"
Set Padding 20

Type "garage --version"
Enter
Sleep 1s

Type "garage open gpt2"
Enter
Sleep 3s

Type "garage extract gpt2 --layer 6"
Enter
Sleep 3s

Type 'garage analyze gpt2 --prompt "The meaning of life"'
Enter
Sleep 3s

Type "garage compare gpt2 distilgpt2"
Enter
Sleep 4s

Sleep 2s
