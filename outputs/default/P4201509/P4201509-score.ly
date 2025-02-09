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
            e'16\ppp\< dis'8 f'8 e'4 e'4 d'16 fis'16 f'16 | ees'8 ees'8 ~ ees'16 f'8 ~ f'16 b'16 g'8 ~ g'16 cis'8 f'16 e'16 | cis'1 | b'1\!\mf |

            dis'1 \bar "|" dis'1 \bar "|" dis'1 \bar "|" dis'1 \bar "|" dis'1 \bar "|" dis'1 \bar "|" dis'1 \bar "|" dis'1 \bar "|" fis'1 \bar "|" fis'1 \bar "|" fis'1 \bar "|" fis'1 \bar "|" fis'1 \bar "|" fis'1 \bar "|" fis'1 \bar "|" fis'1 \bar "|" d'1 \bar "|" fis'4 ~ fis'8 ~ fis'16 e'16 fis'2 \bar "|" fis'2 ~ fis'16 e'16 fis'4 ~ fis'8 \bar "|" fis'16 d'16 fis'2 ~ fis'4 ~ fis'16 g'16 \bar "|" fis'8 g'16 fis'16 e'16 fis'4 e'16 fis'16 e'16 fis'8 g'16 fis'16 \bar "|" e'16 fis'8 e'8 ~ e'16 fis'16 a'16 fis'16 g'16 fis'8 g'16 fis'8 ~ fis'16 \bar "|" e'16 fis'8 ~ fis'16 g'16 fis'8 e'16 fis'4 ~ fis'8 g'16 fis'16 \bar "|" fis'16 g'16 fis'16 e'16 fis'8 e'16 fis'8 e'16 fis'16 e'16 fis'8 g'16 e'16 \bar "|" fis'2 ~ fis'4 ~ fis'8 ~ fis'16 e'16 \bar "|" d'16 e'16 fis'16 e'16 cis'16 e'16 fis'4 cis'16 fis'4 ~ fis'16 \bar "|" fis'16 d'8 b'16 g'16 fis'16 e'16 fis'4 g'16 fis'16 g'16 e'8 \bar "|" g'16 fis'16 g'16 e'16 fis'8 d'16 e'16 d'8 e'16 cis'16 fis'16 g'16 e'16 g'16 \bar "|" g'16 fis'16 e'8 fis'8 g'8 e'8 fis'8 ~ fis'16 g'16 e'16 fis'16 \bar "|" e'16 fis'8 e'16 fis'8 a'16 fis'8 ~ fis'16 b'16 fis'16 g'16 e'16 fis'8 \bar "|" a'16 fis'8 e'16 fis'2 ~ fis'8 g'16 fis'16 \bar "|" e'16 fis'16 e'8 fis'16 e'16 fis'16 g'16 fis'2 \bar "|" e'16 g'16 cis'16 fis'16 cis'16 fis'8 e'8 ~ e'16 fis'16 g'16 cis'16 b'16 fis'8 \bar "|" fis'4 e'16 fis'16 e'16 fis'8 cis'16 e'16 fis'16 e'16 fis'16 e'16 fis'16 \bar "|" fis'8 ~ fis'16 a'16 fis'8 ~ fis'16 e'16 fis'2 \bar "|" f'2 ~ f'4 ~ f'16 ees'16 f'8 \bar "|" a'16 g'16 a'2 ~ a'4 ~ a'8 \bar "|" f'2 ~ f'16 ees'16 f'4 ~ f'8 \bar "|" f'4 ~ f'16 ges'16 f'16 ees'16 f'2 \bar "|" f'2 ~ f'8 ~ f'16 ees'16 f'4 \bar "|"

            f'4\mf\> e'4 b'4 a'8 ~ a'16 g'16 | e'4 g'2 ~ g'4 | b'4 dis'2 ~ dis'4 | f'1\!\ppp |

            \bar "|."
            }
            \new Staff = "left" {
            \clef bass
            % 让左手整体音量更低
            \set Staff.midiMinimumVolume = #0.2
            \set Staff.midiMaximumVolume = #0.5
            r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" d1 \bar "|" a8 ~ a16 g16 a2 ~ a4 \bar "|" a4 ~ a8 ~ a16 g16 a2 \bar "|" a4 f16 a4 ~ a16 bes16 a4 ~ a16 \bar "|" a8 ~ a16 g16 bes16 a8 g16 a4 g16 a16 bes16 a16 \bar "|" g8 a16 g16 a16 c16 a8 bes16 a16 bes16 a8 ~ a16 g16 a16 \bar "|" d16 c16 d2 c16 d16 ees16 d8 ees16 \bar "|" c16 d4 ~ d16 c8 ~ c16 d16 ees16 d16 ees16 d16 c16 d16 \bar "|" a2 ~ a8 ~ a16 g16 a4 \bar "|" a16 e16 a8 ~ a16 g16 a8 f16 a8 e16 g16 a8 g16 \bar "|" f16 g16 a16 g16 a16 bes16 a8 bes16 g16 d16 a16 bes16 a8 f16 \bar "|" g16 c16 a8 bes16 c16 bes8 g16 c16 g16 a16 f16 a16 bes16 c16 \bar "|" c16 bes16 c16 bes8 ~ bes16 c16 a8 bes16 c16 a16 bes16 a8 bes16 \bar "|" bes16 a8 bes8 ~ bes16 a16 d16 c16 bes4 ~ bes8 e16 \bar "|" bes4 ~ bes8 a16 c16 bes4 ~ bes16 d16 bes8 \bar "|" a8 ~ a16 g16 a8 g8 a8 g16 a8 ~ a16 bes16 a16 \bar "|" a16 e16 c16 a16 bes16 f16 bes8 a16 f16 a16 bes16 c16 bes8 f16 \bar "|" a16 bes4 a16 bes16 f16 bes8 a16 bes8 a16 bes16 a16 \bar "|" bes8 ~ bes16 d16 bes2 a16 bes8 ~ bes16 \bar "|" bes4 ~ bes16 a16 bes2 ~ bes8 \bar "|" e16 f2 ~ f4 ~ f8 ~ f16 \bar "|" bes8 ~ bes16 a16 bes2 ~ bes4 \bar "|" bes2 a16 bes8 ~ bes16 c16 bes8 ~ bes16 \bar "|" bes2 ~ bes8 ~ bes16 a16 bes4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|"
            \bar "|."
            }
        >>
        \layout {}
        \midi {}
        }
        