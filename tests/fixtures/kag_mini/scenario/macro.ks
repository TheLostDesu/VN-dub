; macro.ks -- name/voice macros (synthetic fixture)
[macro name=nm]
[ch text="【%t】"][r]
[playse storage=%s buf=1 cond="mp.s!=void"]
[endmacro]

[macro name=voi]
[playse storage=%file buf=1]
[endmacro]
