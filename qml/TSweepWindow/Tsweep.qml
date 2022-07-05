// Analysis Panel

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform



ApplicationWindow {

    property double ti: obs.TIs
    property double tstep: .2
    property double tend: .0

    property int npoints: 12

    property int dwait: 5
    property int twait: 10
    property int ewait: 5

    id: root
    width: 500 ; height: 300

    property int tb_width: 50

    ColumnLayout{
        anchors.centerIn: parent
        Text {
            Layout.alignment: Qt.AlignHCenter
            text: qsTr("Temperature Sweep Analysis")
        }

        // Temperature parameters
        RowLayout{
            Layout.alignment: Qt.AlignHCenter
            spacing: 8
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "T Start"
                }

                DoubleField{
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignHCenter
                    id: tifield
                    text: `${ti}`
                }

            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "T Step"
                }
                DoubleField{
                    Layout.alignment: Qt.AlignHCenter
                    width: tb_width
                    low: 0.005
                    high: 10.0
                    prec: 3
                    text: `${tstep}`
                    id: tstepfield
                }
            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "T End"
                }
                DoubleField{
                    Layout.alignment: Qt.AlignHCenter
                    width: tb_width
                    id: tendfield
                    text: `${tend}`
                }
            }
        }

        // Number of datapoints
        ColumnLayout{
            width: parent.width/2
            Text{
                Layout.alignment: Qt.AlignHCenter
                text: "Number of datapoints for every temperature"
            }

            IntField{
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignHCenter
                id: dpoints
                text: `${npoints}`
            }
        }


        // Waiting times
        Text{
            Layout.alignment: Qt.AlignHCenter
            text: "Waiting Times"
        }
        RowLayout{
            Layout.alignment: Qt.AlignHCenter
            spacing: 8
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "Between datapoints"
                }

                IntField{
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignHCenter
                    id: dpointwait
                    text: `${dwait}`
                    low: 1
                }
            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "Between Temperatures"
                }
                IntField{
                    Layout.alignment: Qt.AlignHCenter
                    id: tstepwait
                    text: `${twait}`
                }
            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "Wait for equilibration"
                }
                IntField{
                    Layout.alignment: Qt.AlignHCenter
                    id: equiwait
                    text: `${ewait}`
                }
            }
        }
        ColumnLayout{
            Layout.alignment: Qt.AlignHCenter
            spacing: 1

            Text{
                Layout.alignment: Qt.AlignHCenter
                text: "Save Image"
            }
            Switch{
                Layout.alignment: Qt.AlignHCenter
                checked: true
                id: saveImg
            }
        }

        Button{
            Layout.alignment: Qt.AlignHCenter
            text: "Start Analysis"
            onClicked: {
                Julia.tempSweepQML(tifield.val, tendfield.val, tstepfield.val, dpoints.val, dpointwait.val, tstepwait.val, equiwait.val, saveImg.checked)
                obs.analysisRunning = true
                Julia.println(obs.analysisRunning)
                close()
            }
        }
    }

}

//     function custIntValidator(text, low, high)
//     {
//         var newText = ""
//         if(isNaN(parseInt(text)))
//     {
//         var newInt = Math.max(0, low)
//         newText = `${newInt}`
//         return newText
//     }
//     if(parseInt(text) < low)
// {
//     newText = `${low}`
//     return newText
// }
// else if(parseInt(text) > high)
// {
//     newText = `${high}`
//     return newText
// }

// return text
// }


// function custDoubleValidator(text, low, high)
// {
//     var newText = ""
//     if(isNaN(parseInt(text)))
// {
//     newText = "0.0"
//     return newText
// }
// if(parseInt(text) < low)
// {
// newText = `${low}`
// return newText
// }
// else if(parseInt(text) > high)
// {
//     newText = `${high}`
//     return newText
// }

// return text
// }


