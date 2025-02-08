
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
            fis'16\ppp\< f'8 a'8 fis'4 ~ fis'16 fis'4 ~ fis'16 bes'16 | g'16 a'8 e'4 ~ e'16 g'4 ~ g'16 a'8 ees'16 | b'4 aes'8 ~ aes'16 b'8 aes'16 fis'4 fis'16 ees'16 | bes'16 e'8 ~ e'16 c'8 ~ c'16 c'8 fis'16 a'8 ~ a'16 fis'8 fis'16\!\mf |

            fis'16 b'8 fis'16 e'8 fis'16 b'16 g'8 fis'8 ~ fis'16 g'16 b'16 e'16 \bar "|" a'16 d'16 b'16 fis'16 g'16 fis'8 b'16 fis'16 e'16 b'16 fis'16 g'16 b'8 fis'16 \bar "|" fis'8 ~ fis'16 d'16 g'16 fis'16 g'16 d'16 fis'8 ~ fis'16 b'8 fis'16 b'8 \bar "|" fis'8 e'16 b'16 fis'16 g'16 fis'16 g'16 fis'4 b'16 e'16 b'8 \bar "|" fis'16 e'16 b'16 fis'4 ~ fis'16 g'16 b'16 g'16 b'16 fis'16 b'16 fis'8 \bar "|" fis'8 b'16 fis'8 e'16 fis'8 g'16 fis'4 a'16 fis'8 \bar "|" fis'2 b'8 ~ b'16 a'16 g'16 b'16 fis'8 \bar "|" ees'4 ~ ees'16 f'16 ees'8 g'16 ees'16 c'16 ees'4 ~ ees'16 \bar "|" cis'4 ~ cis'16 b'16 cis'8 b'16 cis'8 b'16 cis'16 ais'16 cis'8 \bar "|" b'16 cis'8 b'16 dis'8 cis'4 ~ cis'16 b'16 cis'8 b'16 cis'16 \bar "|" f'16 ees'16 des'16 f'8 ees'4 ~ ees'8 ~ ees'16 aes'16 ees'16 des'8 \bar "|" fis'4 ~ fis'16 g'16 fis'8 g'16 fis'16 e'16 fis'16 g'16 e'16 fis'8 \bar "|" aes'8 g'4 ees'16 aes'16 g'4 f'16 aes'16 g'8 \bar "|" g'16 aes'16 g'8 f'16 aes'4 g'8 ~ g'16 aes'16 g'8 aes'16 \bar "|" g'8 ~ g'16 aes'16 g'4 aes'16 ees'16 bes'16 aes'16 g'8 aes'16 f'16 \bar "|" g'16 f'16 g'8 aes'16 g'16 ees'16 aes'16 f'16 g'16 f'8 g'4 \bar "|" f'8 ees'16 f'4 ~ f'16 ees'16 f'4 ges'16 des'16 f'16 \bar "|" ees'16 ges'16 ees'8 f'8 ges'16 f'16 ges'16 bes'16 ees'16 aes'16 f'8 ~ f'16 ges'16 \bar "|" a'8 bes'16 a'8 ~ a'16 bes'16 a'16 bes'8 a'16 g'8 ~ g'16 a'8 \bar "|" a'4 bes'16 a'8 g'16 bes'16 a'8 c'16 g'16 a'8 bes'16 \bar "|" c'16 a'8 bes'16 f'16 c'16 a'8 bes'16 f'16 bes'8 a'4 \bar "|" f'8 ~ f'16 ees'16 f'8 ges'16 aes'16 ees'16 f'16 ges'16 f'4 bes'16 \bar "|" fis'16 g'16 fis'16 b'16 fis'4 g'16 b'8 g'16 fis'16 b'16 fis'8 \bar "|" b'16 g'16 fis'16 g'16 a'8 ~ a'16 g'16 fis'16 g'16 fis'8 g'16 a'16 fis'16 g'16 \bar "|" g'8 f'16 aes'16 g'16 f'8 g'16 f'16 g'8 aes'16 g'16 bes'16 aes'16 g'16 \bar "|" a'16 g'4 c'8 g'16 bes'16 a'8 g'16 bes'16 c'16 g'16 a'16 \bar "|" a'16 g'16 a'8 bes'16 g'16 a'16 g'8 ~ g'16 a'16 bes'8 g'16 c'8 \bar "|" a'16 g'8 c'16 a'16 bes'8 a'8 ~ a'16 c'16 a'8 ~ a'16 bes'16 g'16 \bar "|" f'16 g'8 aes'16 g'16 aes'16 g'8 f'16 g'4 ~ g'16 aes'16 g'16 \bar "|" fis'1 \bar "|"

            fis'1\mf\> | fis'2 ~ fis'8 c'4 ~ c'8 | ees'1 | b'1\!\ppp |

            \bar "|."
            }
            \new Staff = "left" {
            \clef bass
            % 让左手整体音量更低
            \set Staff.midiMinimumVolume = #0.2
            \set Staff.midiMaximumVolume = #0.5
            r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" a8 d8 g16 bes16 a16 d16 a16 d16 bes16 a16 g16 a16 bes16 g16 \bar "|" f16 c16 a8 ~ a16 d16 a8 g16 d8 bes8 a16 d8 \bar "|" bes16 a16 d8 bes16 a16 f16 a8 d16 a8 ~ a16 d16 f16 a16 \bar "|" bes16 d16 a8 ~ a16 d8 a8 ~ a16 g8 a16 bes16 a16 d16 \bar "|" a8 bes16 a16 bes16 a4 d8 g16 d16 a8 d16 \bar "|" a8 ~ a16 bes16 a8 d16 c16 g16 a4 ~ a8 ~ a16 \bar "|" c4 f16 c16 ees16 f8 c16 f16 c4 des16 \bar "|" e4 ~ e16 c16 g16 f16 e2 \bar "|" c8 ~ c16 bes16 c4 ~ c16 bes16 c4 aes16 bes16 \bar "|" f16 aes16 g8 ~ g16 f16 g8 aes16 g16 f16 g8 ~ g16 f16 g16 \bar "|" g4 c16 g16 aes16 g16 aes16 f16 aes16 g8 ~ g16 f8 \bar "|" e16 f16 e8 f16 e8 d8 e4 ~ e8 f16 \bar "|" a16 bes16 c8 bes16 g16 c16 bes16 c16 bes4 ~ bes8 ~ bes16 \bar "|" bes4 c8 bes16 c16 bes8 c16 a16 c8 bes16 c16 \bar "|" bes16 c16 bes16 d16 bes4 ~ bes8 c8 g16 c16 a16 bes16 \bar "|" a16 bes8 a8 bes8 ~ bes16 c16 bes8 ~ bes16 a16 c16 bes16 g16 \bar "|" bes16 g16 bes8 c16 bes2 a16 bes16 a16 \bar "|" bes16 d16 bes8 c16 bes16 c8 bes16 c16 e16 a4 bes16 \bar "|" bes16 c16 bes8 ~ bes16 a16 bes4 ~ bes16 c16 a8 c8 \bar "|" bes16 c16 a16 c16 bes8 d16 bes8 ~ bes16 a16 bes8 ~ bes16 c16 bes16 \bar "|" bes16 c16 g16 bes16 d16 bes8 c16 bes8 ~ bes16 c8 g16 bes16 d16 \bar "|" bes16 c8 ~ c16 des16 c16 des16 c8 ~ c16 f16 c16 ees16 bes16 c8 \bar "|" bes8 c16 bes16 c8 bes16 e8 bes16 e16 bes8 e16 bes8 \bar "|" c8 d16 bes8 d16 bes16 c16 d16 c16 bes16 e16 bes16 c8 d16 \bar "|" c16 bes8 d16 bes16 c16 a8 bes16 a16 bes8 a16 c16 bes8 \bar "|" a16 d16 bes8 a8 bes16 a16 c16 bes16 d16 a16 c16 a16 d16 a16 \bar "|" a8 bes8 c16 a16 bes16 d16 a16 c16 bes16 a16 c16 bes16 d16 a16 \bar "|" d16 a16 c16 a16 d16 c16 bes16 c16 bes8 ~ bes16 a16 bes4 \bar "|" bes16 c16 bes8 c8 bes16 a16 bes16 a16 bes4 ~ bes8 \bar "|" bes1 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|" r4 r4 r4 r4 \bar "|"
            \bar "|."
            }
        >>
        \layout {}
        \midi {}
        }
        