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
            r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" a8 c16 a16 bes16 d16 c16 d8 c8 d16 c8 f16 g16 \bar "|" a16 g8 a16 e16 d16 bes16 f16 d16 g16 f16 d16 e16 f16 d16 c16 \bar "|" g16 c16 bes16 a16 bes16 g8 d8 c16 a16 e16 f8 bes16 e16 \bar "|" g16 a16 e16 d16 a16 bes8 g16 c16 a16 e16 d16 f16 c16 bes16 c16 \bar "|" bes8 aes16 c16 des16 f8 c16 f8 bes16 g16 f16 ees16 f16 c16 \bar "|" c16 a16 bes16 g16 a8 f16 e16 bes8 a16 g16 f16 a8 bes16 \bar "|" c16 g16 ees16 f16 d16 a16 g16 f16 g8 d16 g16 ees16 c8 ~ c16 \bar "|" a16 c16 bes16 e16 d16 e16 d16 f8 d16 a16 e16 d16 c16 d16 f16 \bar "|" e16 g16 e16 bes16 g16 d16 bes16 a16 c16 d16 a4 ~ a16 g16 \bar "|" aes8 bes16 f16 g16 des16 f16 ees8 f16 bes16 g16 ees16 c16 f16 bes16 \bar "|" d8 bes16 g16 d16 bes16 ees16 f16 ees16 d16 g16 d16 c16 ees16 g16 c16 \bar "|" c16 des16 c16 bes8 f16 bes16 g16 f16 aes16 f16 bes16 g16 f8 ~ f16 \bar "|" f16 c16 g16 f16 g8 bes16 a16 c16 bes16 g8 ~ g16 d16 g8 \bar "|" aes16 des16 f16 g16 f16 aes16 g16 f16 aes8 bes16 ees16 des8 aes8 \bar "|" g16 ees8 bes16 c16 g8 bes8 ees16 c16 f16 a16 d16 ees16 g16 \bar "|" f8 d16 bes16 c16 a16 g16 f16 ees16 bes16 c16 g16 f16 ees16 a8 \bar "|" g8 f16 c16 d16 f16 ees16 c16 ees8 d16 a16 c16 ees16 g16 f16 \bar "|" bes16 c16 g16 ees16 f16 c16 d16 bes16 c16 a16 ees16 g16 bes8 ees16 c16 \bar "|" ees16 g16 a16 f16 g8 c16 a16 ees16 c16 g8 a16 g16 c16 ees16 \bar "|" g16 c16 f8 d16 g16 ees16 d8 f16 ees8 a16 c16 g16 bes16 \bar "|" bes16 ees16 f4 ~ f16 a16 f16 g16 c8 g16 ees16 c16 ees16 \bar "|" b16 c16 e8 b16 c16 f16 c16 a16 g16 b16 c16 f16 c16 g8 \bar "|" c16 d16 e16 g16 c16 b16 e16 g16 f16 a16 f16 c16 d16 g16 c16 d16 \bar "|" c16 f8 ees16 g16 c16 g16 bes8 ees16 f16 des16 ees16 des16 f16 c16 \bar "|" c16 f16 bes16 d16 g16 d16 ees16 g16 bes8 a16 d8 g16 ees16 d16 \bar "|" b16 a16 e16 d16 g16 e16 f8 b16 a16 c16 a8 d16 e16 g16 \bar "|" g16 ees16 d16 bes8 f16 bes8 d8 f8 g8 ees16 c16 \bar "|" e16 f16 a16 b16 a8 g16 a16 c16 f16 c16 g16 c8 e16 c16 \bar "|" g16 f16 d16 g16 f16 a16 c16 b16 a16 f8 d16 b16 c16 b16 c16 \bar "|" b16 c16 a8 e16 f16 g16 b8 f16 a16 c8 f16 d16 e16 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|"
            \bar "|."
            }
        >>
        \layout {}
        \midi {}
        }
        