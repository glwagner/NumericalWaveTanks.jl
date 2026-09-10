import base64, re
F='/u/glwagner/NumericalWaveTanks.jl/figures/anti_stokes/'
s=open(F+'gallery.html').read()
# keep from <title> to </main>
s=s[s.find('<title>'):s.rfind('</main>')+len('</main>')]
def b64(name, mime): return f"data:{mime};base64," + base64.b64encode(open(F+name,'rb').read()).decode()
def sub_video(m):
    name=m.group(1).replace('_small.mp4','_web.mp4'); return 'src="'+b64(name,'video/mp4')+'"'
def sub_img(m): return 'src="'+b64(m.group(1),'image/png')+'"'
s=re.sub(r'src="https://cdn\.jsdelivr\.net/gh/glwagner/NumericalWaveTanks\.jl@glw/anti-stokes/figures/anti_stokes/([^"]+\.mp4)"', sub_video, s)
s=re.sub(r'src="https://raw\.githubusercontent\.com/glwagner/NumericalWaveTanks\.jl/glw/anti-stokes/figures/anti_stokes/([^"]+\.png)"', sub_img, s)
# theme tokens: explicit-choice guards in addition to the system media query
s=s.replace('@media (prefers-color-scheme: dark) { :root { --bg: #0E1620; --ink: #E6EDF3; --muted: #97A6B4; --accent: #6EA8E8; --rule: #26323F; } }',
 '@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #0E1620; --ink: #E6EDF3; --muted: #97A6B4; --accent: #6EA8E8; --rule: #26323F; } }\n:root[data-theme="dark"] { --bg: #0E1620; --ink: #E6EDF3; --muted: #97A6B4; --accent: #6EA8E8; --rule: #26323F; }')
s=s.replace('a { color: var(--accent); }', 'a { color: var(--accent); }\na:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }')
assert 'jsdelivr' not in s and 'raw.githubusercontent' not in s
out='/tmp/claude-94375/-u-glwagner-NumericalWaveTanks-jl/7859c402-c537-4289-a03d-b2fa7f9ffe85/scratchpad/gallery_artifact.html'
open(out,'w').write(s); print(len(s)/1e6,'MB', s.count('<video'), 'videos', s.count('<img'), 'images')
