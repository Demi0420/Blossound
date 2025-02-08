
        \version "2.22.1"
        \header {
        title = "Sheet Music"
        composer = "Yao."
        }

        \score {
        % 使用钢琴连谱号 (PianoStaff)
        \new PianoStaff <<
            \new Staff = "right" {
            \clef treble
            fis'16\ppp\< f'8 b'8 fis'4 ~ fis'16 fis'4 ~ fis'16 bes'16 | g'16 b'8 c'4 ~ c'16 g'4 ~ g'16 b'8 ees'16 | a'4 aes'8 ~ aes'16 a'8 aes'16 fis'4 fis'16 ees'16 | bes'16 c'8 ~ c'16 e'8 ~ e'16 e'8 fis'16 b'8 ~ b'16 fis'8 fis'16\!\mf |

            g'16 fis'16 b'16 fis'16 e'16 fis'16 e'16 g'16 e'16 fis'16 b'8 fis'16 g'16 fis'16 b'16 \bar "|" b'8 e'16 b'16 g'16 fis'8 g'16 d'16 fis'4 b'16 a'16 b'16 \bar "|" fis'16 g'16 d'16 fis'16 g'16 fis'4 b'8 fis'16 b'16 d'16 fis'16 b'16 \bar "|" b'8 g'16 fis'16 g'16 e'16 b'16 fis'4 ~ fis'8 e'16 b'16 fis'16 \bar "|" fis'16 b'16 e'16 fis'4 ~ fis'16 b'16 fis'16 g'16 fis'16 b'8 fis'16 g'16 \bar "|" fis'16 e'16 fis'4 a'16 fis'4 g'16 fis'8 b'16 fis'16 \bar "|" fis'8 g'16 fis'16 b'8 fis'8 b'16 fis'8 a'16 fis'8 b'16 fis'16 \bar "|" g'16 f'16 ees'4 ~ ees'16 c'16 ees'2 \bar "|" cis'8 b'16 cis'16 b'16 cis'8 ais'16 b'16 cis'4 ~ cis'8 ~ cis'16 \bar "|" cis'16 b'16 cis'16 dis'16 cis'16 b'16 cis'16 b'8 cis'8 ~ cis'16 dis'16 cis'8 ~ cis'16 \bar "|" des'16 ees'16 f'8 ees'4 ~ ees'16 f'16 des'16 ees'8 des'16 ees'16 aes'16 \bar "|" fis'16 e'16 fis'4 g'16 fis'4 ~ fis'16 g'16 e'16 g'16 fis'16 \bar "|" g'8 aes'16 g'16 ees'16 f'16 aes'8 ~ aes'16 g'4 ~ g'8 ~ g'16 \bar "|" g'16 aes'16 g'16 aes'8 f'16 g'8 aes'16 g'16 aes'16 g'16 aes'16 g'16 aes'16 g'16 \bar "|" f'16 bes'16 aes'16 ees'16 g'4 aes'8 g'8 aes'16 g'8 ~ g'16 \bar "|" g'16 f'16 g'16 f'16 g'8 ees'16 g'16 aes'16 f'16 g'8 aes'16 f'16 g'8 \bar "|" des'16 f'4 ~ f'8 ges'16 f'8 ees'16 f'16 ees'16 f'8 ~ f'16 \bar "|" aes'16 f'16 bes'16 ges'8 f'16 ees'8 f'16 ges'16 ees'16 f'16 ges'16 f'8 ees'16 \bar "|" a'8 bes'16 g'16 a'16 g'16 bes'16 g'16 a'4 ~ a'16 bes'8 a'16 \bar "|" a'8 g'16 bes'16 a'4 ~ a'16 bes'8 g'16 c'16 a'8 ~ a'16 \bar "|" a'4 c'16 bes'16 c'16 bes'16 a'16 f'16 a'8 f'16 bes'16 a'16 bes'16 \bar "|" bes'16 ees'8 f'4 ges'16 f'8 ~ f'16 ges'16 aes'16 f'8 ~ f'16 \bar "|" fis'16 b'16 fis'8 g'16 fis'4 b'8 fis'16 g'16 b'16 g'16 fis'16 \bar "|" g'8 a'16 g'16 a'16 fis'16 b'16 fis'8 a'16 fis'16 g'16 fis'16 g'8 a'16 \bar "|" f'16 bes'16 g'8 aes'16 g'16 f'16 g'16 aes'16 g'16 f'16 g'8 ~ g'16 f'16 aes'16 \bar "|" c'16 a'16 c'16 g'16 bes'16 g'8 ~ g'16 c'16 bes'16 g'16 a'16 g'8 a'8 \bar "|" c'16 a'16 bes'8 c'16 g'16 a'16 g'16 a'16 g'8 bes'16 g'16 a'16 g'16 a'16 \bar "|" a'8 g'16 c'16 a'8 bes'16 a'16 g'16 a'16 bes'16 g'16 c'16 bes'16 a'8 \bar "|" g'16 aes'16 f'16 g'16 f'16 g'8 ~ g'16 aes'16 g'16 aes'16 g'4 ~ g'16 \bar "|" fis'1 \bar "|"

            fis'1\mf\> | fis'2 ~ fis'8 e'4 ~ e'8 | ees'1 | a'1\!\ppp |

            \bar "|."
            }
            \new Staff = "left" {
            \clef bass
            % 让左手整体音量更低
            \set Staff.midiMinimumVolume = #0.2
            \set Staff.midiMaximumVolume = #0.5
            r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" a16 d16 g16 d16 g16 a16 bes16 a16 d16 bes16 g16 a8 d16 a16 bes16 \bar "|" d8 f16 a8 bes16 c16 a16 g16 bes16 a8 d8 ~ d16 a16 \bar "|" d16 a8 d16 f16 bes16 d16 a16 f16 a8 ~ a16 bes16 a16 d16 a16 \bar "|" a16 g16 d8 a16 bes16 a8 ~ a16 d16 a16 bes16 d16 g16 a8 \bar "|" d16 a8 d16 bes16 a16 bes16 a16 d16 g16 a8 ~ a16 d16 a8 \bar "|" a8 bes16 a8 g16 a4 ~ a8 d16 a16 c16 a16 \bar "|" c8 f16 c16 f16 c8 ees16 c8 f16 c8 des16 c16 f16 \bar "|" e2 ~ e4 c16 g16 e16 f16 \bar "|" c4 ~ c16 bes16 c8 aes16 bes16 c16 bes16 c4 \bar "|" f8 aes8 f16 g8 ~ g16 f16 g4 ~ g8 ~ g16 \bar "|" f16 g16 f16 g8 ~ g16 aes16 g16 c16 g16 aes8 f16 g8 ~ g16 \bar "|" e2 ~ e16 d16 f16 d16 e8 f8 \bar "|" bes16 c8 bes16 c16 bes16 g16 bes8 a16 bes4 ~ bes16 c16 \bar "|" bes16 c16 bes16 c16 bes8 ~ bes16 c4 bes16 a16 bes16 c16 bes16 \bar "|" c16 bes8 c8 bes16 a16 d16 bes8 ~ bes16 g16 c16 bes8 ~ bes16 \bar "|" bes8 a16 c16 bes16 g16 a16 c16 bes8 ~ bes16 a16 bes8 a16 bes16 \bar "|" bes4 a16 g16 bes16 a16 c16 bes4 ~ bes8 ~ bes16 \bar "|" e16 c8 bes16 a8 ~ a16 bes8 c16 d16 bes8 a16 c16 bes16 \bar "|" bes16 c16 a16 c16 bes16 a8 bes4 ~ bes8 c8 bes16 \bar "|" d16 c16 bes16 a16 bes8 c16 bes8 ~ bes16 a16 bes8 ~ bes16 c16 bes16 \bar "|" bes16 g16 bes16 c16 bes16 d8 bes8 ~ bes16 g16 bes8 c8 ~ c16 \bar "|" f16 c4 ees16 c8 des16 c8 des16 bes16 c16 bes16 c16 \bar "|" bes4 ~ bes16 c16 bes16 e16 bes16 e16 c16 bes16 e16 c16 bes16 e16 \bar "|" d16 e16 bes8 c16 d16 c16 d16 c8 d16 bes16 c16 bes16 c16 bes16 \bar "|" a16 bes16 c16 bes8 ~ bes16 a16 bes16 c16 a16 bes8 c16 a16 d16 bes16 \bar "|" a16 c16 a16 bes8 a16 c16 a16 d8 a16 d16 a16 bes16 a16 bes16 \bar "|" bes16 d16 a16 c16 a16 c16 a16 bes16 c16 d16 a16 bes8 ~ bes16 a8 \bar "|" c16 bes4 d16 bes8 ~ bes16 c16 a8 d16 bes16 c16 a16 \bar "|" bes8 c16 bes4 a16 bes4 a16 bes16 c8 \bar "|" bes1 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|"
            \bar "|."
            }
        >>
        \layout {}
        \midi {}
        }
        