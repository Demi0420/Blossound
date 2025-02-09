\version "2.24.1"
        \header {
        title = "Music Score"
        % composer = "Yao."
        }

        \score {
        % 使用钢琴连谱号 (PianoStaff)
        \new PianoStaff <<
            \new Staff = "right" {
            \clef treble
            d'16\ppp\< c'8 b'8 d'4 d'4 cis'16 fis'16 b'16 | ees'8 ees'8 ~ ees'16 b'8 ~ b'16 g'16 gis'8 ~ gis'16 e'8 b'16 d'16 | e'1 | g'1\!\mf |

            fis'1 \bar "|" d'1 \bar "|" d'1 \bar "|" d'1 \bar "|" d'2 ~ d'4 c'16 d'8 c'16 \bar "|" e'16 b'8 fis'16 gis'16 fis'16 gis'16 e'16 b'16 gis'16 b'16 fis'8 e'16 gis'16 fis'16 \bar "|" d'2 ~ d'4 ~ d'16 g'16 d'8 \bar "|" fis'1 \bar "|" d'1 \bar "|" d'16 c'16 a'16 d'8 ~ d'16 a'16 d'8 ~ d'16 c'16 d'8 ~ d'16 ees'16 d'16 \bar "|" e'16 gis'16 e'16 gis'16 b'16 fis'16 e'16 b'16 fis'8 ~ fis'16 b'4 e'16 \bar "|" e'16 fis'16 gis'16 b'16 fis'8 e'16 gis'8 ~ gis'16 fis'16 gis'16 b'16 fis'16 b'16 fis'16 \bar "|" c'4 bes'16 c'8 d'16 c'16 f'16 c'16 d'16 bes'8 c'16 f'16 \bar "|" d'4 ~ d'8 a'16 d'8 ees'16 d'16 bes'16 d'16 ees'16 d'8 \bar "|" fis'1 \bar "|" d'1 \bar "|" fis'16 gis'16 cis'16 gis'16 fis'16 dis'16 fis'16 e'16 gis'16 cis'16 ais'16 cis'8 fis'16 cis'8 \bar "|" fis'4 b'16 fis'2 ~ fis'8 ~ fis'16 \bar "|" cis'16 fis'16 gis'16 fis'16 ais'16 gis'16 fis'8 gis'16 b'8 fis'4 ~ fis'16 \bar "|" ais'16 gis'8 fis'16 cis'16 gis'16 ais'16 gis'8 ~ gis'16 ais'16 gis'16 fis'16 cis'16 ais'16 gis'16 \bar "|" g'16 d'2 ~ d'8 ~ d'16 ees'16 d'8 ~ d'16 \bar "|" fis'1 \bar "|" ees'16 d'4 c'16 g'16 d'4 a'16 g'16 d'8 ~ d'16 \bar "|" fis'8 ~ fis'16 b'16 ais'16 b'16 fis'8 cis'16 fis'16 gis'16 fis'16 b'16 gis'16 fis'8 \bar "|" d'1 \bar "|" fis'8 ~ fis'16 ais'16 fis'8 ~ fis'16 e'16 b'16 fis'8 ~ fis'16 gis'16 fis'8 ~ fis'16 \bar "|" d'2 ~ d'8 ~ d'16 c'16 d'4 \bar "|" fis'1 \bar "|" fis'1 \bar "|" b'8 fis'8 ~ fis'16 b'8 fis'16 b'16 gis'8 b'16 gis'16 e'16 b'16 fis'16 \bar "|" d'2 ~ d'4 c'16 d'16 ees'16 d'16 \bar "|" fis'1 \bar "|" fis'1 \bar "|" fis'1 \bar "|" fis'1 \bar "|"

            fis'1\mf\> | d'2 g'2 | ais'2 ~ ais'4 gis'4 | d'1\!\ppp |

            \bar "|."
            }
            \new Staff = "left" {
            \clef bass
            % 让左手整体音量更低
            \set Staff.midiMinimumVolume = #0.2
            \set Staff.midiMaximumVolume = #0.5
            r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" a1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" c16 d4 ~ d8 c16 d2 \bar "|" g16 d8 c16 ees16 d16 c8 ees16 g16 d16 g16 ees8 g16 d16 \bar "|" d2 ~ d4 g16 d8 ~ d16 \bar "|" d1 \bar "|" d1 \bar "|" d8 c16 a16 c16 d8 ees16 d4 ~ d8 a16 d16 \bar "|" ees16 c16 g16 c16 d8 g4 c16 d8 c16 g16 ees16 \bar "|" g16 ees16 d16 ees16 c16 ees16 d16 ees8 d16 g16 c16 g16 d8 ~ d16 \bar "|" g8 f16 g4 c16 f16 a16 g16 c16 g16 a16 f16 g16 \bar "|" d8 bes16 ees16 d16 ees16 d8 ~ d16 a16 d4 ~ d8 \bar "|" d1 \bar "|" d1 \bar "|" a16 g16 d8 ~ d16 b16 d16 g16 d16 f16 a16 d16 g8 e16 a16 \bar "|" d2 ~ d4 ~ d8 g16 d16 \bar "|" a16 d16 g16 b16 a8 g8 c16 g4 c16 g8 \bar "|" g16 f16 g16 a8 g16 a16 g16 c16 f16 g4 c16 a16 \bar "|" ees16 d4 ~ d8 ~ d16 g16 d4 ~ d8 ~ d16 \bar "|" a1 \bar "|" d16 c16 d8 ~ d16 g16 ees16 d8 g16 d16 a16 d4 \bar "|" g16 f16 d8 g16 ees16 d16 ees16 d16 g16 d4 a16 d16 \bar "|" d1 \bar "|" d4 g16 ees16 d8 ~ d16 f16 d8 c16 d8 ~ d16 \bar "|" d4 c16 d2 ~ d8 ~ d16 \bar "|" a1 \bar "|" a1 \bar "|" a16 d16 a16 d16 a8 bes16 d8 bes8 d16 g16 d8 a16 \bar "|" d8 ~ d16 c16 d8 ees16 d2 ~ d16 \bar "|" a1 \bar "|" a1 \bar "|" a1 \bar "|" f1 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|"
            \bar "|."
            }
        >>
        \layout {}
        \midi {}
        }
        